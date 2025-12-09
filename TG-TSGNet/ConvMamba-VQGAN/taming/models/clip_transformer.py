import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from main import instantiate_from_config
from taming.modules.util import SOSProvider
from CLIP import clip

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class CLIPCond(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.init_first_stage_from_ckpt(first_stage_config)
        
        permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    

    def forward(self, imgs, text=None):
        _, z_indices = self.encode_to_z(imgs)
        if text is None:
            clip_vectors = self.encode_imgs_to_c(imgs)

        else:
            clip_vectors = self.encode_text_to_c(text)

        clip_vectors = clip_vectors.unsqueeze(dim=1)
        
        if text is None and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(a_indices[:, :-1], embeddings=clip_vectors)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_imgs_to_c(self, imgs):
        imgs = torch.stack([self.clip_preprocess(transforms.ToPILImage()(img)) for img in imgs], dim=0).cuda()
        return self.clip_model.encode_image(imgs).float()
        
    @torch.no_grad()
    def encode_text_to_c(self, text):
        text = clip.tokenize (text).cuda()
        return self.clip_model.encode_text(text).float()

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x
        
    def logits_to_z(self, logits):
    # logits, probs: size (B, L, vocab_size)
        probs = F.softmax(logits, dim=-1)
        embedding = self.first_stage_model.quantize.embedding.weight  # (vocab_size, E)
        vocab_size, E = embedding.shape

        # argmax:  size (B, L)
        # one-hot: size (B, L, vocab_size)
        argmax = torch.argmax(logits, dim=2)
        onehot = F.one_hot(argmax, num_classes=vocab_size).float()

        # quantize
        onehot = probs + (onehot - probs).detach()

        B, L, vocab_size = onehot.shape
        z = onehot.view(B * L, vocab_size) @ embedding
        z = z.view(B, L, E)                # ([16, 16, 256])
        # print("zz shape:", z.shape)

        return z
    
    
    def shared_step(self, batch, batch_idx):
        x = batch['image']
        x = torch.permute(x, (0, 3, 1, 2))
        logits, target = self(x)
        original_c = self.encode_imgs_to_c(x)
        # print("logits:", logits.shape)
        
        
        z = self.logits_to_z(logits).view(-1, 4, 4, 256) # ([1, 16, 16, 256])
        #print("z1 shape:", z.shape)
        z = z.permute(0, 3, 1, 2)                          # ([1, 256, 16, 16])
        # print("z2 shape:", z.shape)
        
        new_x = self.first_stage_model.decode(z)  # [B, 3, H, W]
        # print("new_x shape:", new_x.shape)
        
        clip_x_recon = F.interpolate(new_x, size=224, mode='bilinear')
        
        # new_x = self.decode_to_img(new_x, x.shape)
        # new_x = self.decode_to_img(logits, x.shape) Noooooo
        
        new_c = self.encode_imgs_to_c(clip_x_recon)
        
        # print("new_c shape:", new_c.shape)
        # print("original_c shape:", original_c.shape)

        loss_clip = F.cosine_embedding_loss(original_c, new_c, target)
        # loss_clip = F.mse_loss(original_c, new_c)
        loss_transformer = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
        clip_factor=0.2 #following CLIP-GEN paper
        loss = loss_clip*clip_factor + loss_transformer
        return loss_clip, loss_transformer, loss
        

    # def training_step(self, batch, batch_idx):
        # loss = self.shared_step(batch, batch_idx)
        # self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # return loss
        
    def training_step(self, batch, batch_idx):
        loss_clip, loss_transformer, loss = self.shared_step(batch, batch_idx)
        self.log("train/loss_clip", loss_clip, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_transformer", loss_transformer, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
        # loss = self.shared_step(batch, batch_idx)
        # self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # return loss
        
    def validation_step(self, batch, batch_idx):
        loss_clip, loss_transformer, loss = self.shared_step(batch, batch_idx)
        self.log("val/loss_clip", loss_clip, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_transformer", loss_transformer, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
