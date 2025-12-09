import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision

from thop import profile


from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

from taming.data.utils import custom_collate
import pytorch_lightning as pl
import torch.nn as nn
import utils
import os
import yaml
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sr import SRNO, SharedLayers
from datasets.image_folder import ImageFolder
from datasets.wrappers import SRImplicitDownsampledFast
from tqdm import tqdm
from functools import partial

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

from taming.models.vqgan_sr import VQModel

from pytorch_lightning.trainer.supporters import CombinedLoader

import warnings


warnings.filterwarnings("ignore")




def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate, drop_last=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size*4,
                          num_workers=self.num_workers, collate_fn=custom_collate, drop_last=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency_train, batch_frequency_val, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq_train = batch_frequency_train
        self.batch_freq_val = batch_frequency_val
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps_train = [2 ** n for n in range(int(np.log2(self.batch_freq_train)) + 1)]
        if not increase_log_steps:
            self.log_steps_train = [self.batch_freq_train]
        self.clamp = clamp
        
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.batch_freq_val)) + 1)]
        if not increase_log_steps:
            self.log_steps_val = [self.batch_freq_val]
        self.clamp = clamp
        
    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img_train(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency_train(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()
                
    def log_img_val(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency_val(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency_train(self, batch_idx):
        if (batch_idx % self.batch_freq_train) == 0 or (batch_idx in self.log_steps_train):
            try:
                self.log_steps_train.pop(0)
            except IndexError:
                pass
            return True
        return False
    
    def check_frequency_val(self, batch_idx):
        if (batch_idx % self.batch_freq_val) == 0 or (batch_idx in self.log_steps_val):
            try:
                self.log_steps_val.pop(0)
            except IndexError:
                pass
            return True
        return False
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img_train(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img_val(pl_module, batch, batch_idx, split="val")


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)



def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False, mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        c = 1 if not mcell else max(scale / scale_max, 1)

        with torch.no_grad():
            pred = model(inp, coord, cell * c)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        res = metric_fn(pred, batch['gt'])

        val_res.add(res.item(), inp.shape[0])
        pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


class SRModel(pl.LightningModule):
    def __init__(self, model, lr=4e-5, warmup_epochs=50, multiplier=10):
        super(SRModel, self).__init__()
        self.model = model
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.multiplier = multiplier
        self.loss_fn = nn.L1Loss()
        self.data_norm = {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]}
        }
        self.epoch_losses = []
        self.epoch_psnrs = []
        self.best_psnr = float('-inf')

    def forward(self, x, coord, cell):
        return self.model(x, coord, cell)

    def training_step(self, batch, batch_idx):
        inp = batch['inp']
        coord = batch['coord']
        cell = batch['cell']
        gt = batch['gt']

        pred = self(inp, coord, cell)
        loss = self.loss_fn(pred, gt)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inp = batch['inp']
        coord = batch['coord']
        cell = batch['cell']
        gt = batch['gt']

        pred = self(inp, coord, cell)
        loss = self.loss_fn(pred, gt)

        psnr = utils.calc_psnr(pred, gt)
                
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)
        return {'val_loss': loss, 'val_psnr': psnr}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.tensor([x['val_loss'] for x in outputs]))
        avg_psnr = torch.mean(torch.tensor([x['val_psnr'] for x in outputs]))

        self.epoch_losses.append(avg_loss.item())
        self.epoch_psnrs.append(avg_psnr.item())
        
        self.log('epoch_loss', avg_loss)
        self.log('epoch_psnr', avg_psnr)
        
        
        epoch_index = self.current_epoch
        with open(os.path.join(self.trainer.log_dir, 'epoch_losses_and_psnrs.yaml'), 'a') as f:
            yaml.dump([{ 
                'epoch': epoch_index,
                'epoch_loss': avg_loss.item(),
                'epoch_psnr': avg_psnr.item()
            }], f, sort_keys=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        warmup_scheduler = GradualWarmupScheduler(optimizer, self.multiplier, self.warmup_epochs, after_scheduler=scheduler)
        return [optimizer], [warmup_scheduler]
    
    def save_weights(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)


class SRDataModule(pl.LightningDataModule):
    def __init__(self, train_root, val_root, batch_size=64, inp_size=64, scale_min=2, scale_max=4, augment=True):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def setup(self, stage=None):
        train_dataset = ImageFolder(
            root_path=self.train_root,
            repeat=10
        )
        self.train_dataset = SRImplicitDownsampledFast(
            dataset=train_dataset,
            inp_size=self.inp_size,
            scale_max=self.scale_max,
            augment=self.augment
        )

        val_dataset = ImageFolder(
            root_path=self.val_root
        )
        self.val_dataset = SRImplicitDownsampledFast(
            dataset=val_dataset,
            scale_min=self.scale_min,
            scale_max=self.scale_min
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


class srvq(pl.LightningModule):
    def __init__(self, sr_model: SRModel, vq_model: VQModel):
        super(srvq, self).__init__()
        self.sr_model = sr_model
        self.vq_model = vq_model

    def training_step(self, batch, batch_idx, optimizer_idx):
    
        # print("***********************************************")
        # print("optimizer_idx:", optimizer_idx)
        # print(f"Global step from training_step: {self.global_step}")
        
        
        
        if self.trainer.current_epoch < 30:

            if optimizer_idx == 1:
                vq_batch = batch[1]
                loss = self.vq_model.training_step(vq_batch, batch_idx, optimizer_idx, self.global_step)
                self.log("vq_train_loss_ae", loss, prog_bar=True)
                return loss
    
            elif optimizer_idx == 2:
                vq_batch = batch[1]
                loss = self.vq_model.training_step(vq_batch, batch_idx, optimizer_idx, self.global_step)
                self.log("vq_train_loss_disc", loss, prog_bar=True)
                return loss
        
        else:
        
            if optimizer_idx == 0:
                
                sr_batch = batch[0]  
                loss = self.sr_model.training_step(sr_batch, batch_idx)
                
                # print("SR Model Loss:", loss)
                
                self.log("sr_train_loss", loss, prog_bar=True)
                
                    
                return loss
                
            elif optimizer_idx == 1:
            
                vq_batch = batch[1] 
                 
                loss = self.vq_model.training_step(vq_batch, batch_idx, optimizer_idx, self.global_step)
                
                # print("VQ Model Loss:", loss)
                
                self.log("vq_train_loss_ae", loss, prog_bar=True)
                
                    
                return loss
                
                
            elif optimizer_idx == 2:
                vq_batch = batch[1] 
                 
                loss = self.vq_model.training_step(vq_batch, batch_idx, optimizer_idx, self.global_step)
                
                # print("VQ Model Loss:", loss)
                
                self.log("vq_train_loss_disc", loss, prog_bar=True)
                
                    
                return loss

    def validation_step(self, batch, batch_idx):

        sr_val_batch = batch[0]
        vq_val_batch = batch[1]

        sr_val_output = self.sr_model.validation_step(sr_val_batch, batch_idx)
        vq_val_output = self.vq_model.validation_step(vq_val_batch, batch_idx, self.global_step)
        

        self.log("sr_val_loss", sr_val_output['val_loss'], prog_bar=True)
        self.log("sr_val_psnr", sr_val_output['val_psnr'], prog_bar=True)
        self.log("vq_val_rec_loss", vq_val_output["rec_loss"], prog_bar=True)
        self.log("vq_val_aeloss", vq_val_output["aeloss"], prog_bar=True)
        
        
        if "log_dict_ae" in vq_val_output:
            self.log_dict(vq_val_output["log_dict_ae"], prog_bar=False)
        if "log_dict_disc" in vq_val_output:
            self.log_dict(vq_val_output["log_dict_disc"], prog_bar=False)
        
        # Save best SR and VQ model weights based on validation metrics
        if self.trainer.global_rank == 0:
            if sr_val_output['val_psnr'] > self.sr_model.best_psnr:
                self.sr_model.best_psnr = sr_val_output['val_psnr']
                self.save_sr_model_weights()

            if vq_val_output["rec_loss"] < self.vq_model.best_rec_loss:
                self.vq_model.best_rec_loss = vq_val_output["rec_loss"]
                self.save_vq_model_weights()

        
        return sr_val_output, vq_val_output
        
    def validation_epoch_end(self, outputs):
    
        if self.trainer.global_rank == 0:
            self.save_sr_epoch_end_weights()
            self.save_vq_epoch_end_weights()

    def configure_optimizers(self):

        sr_optimizers, sr_schedulers = self.sr_model.configure_optimizers()
        vq_optimizers, vq_schedulers = self.vq_model.configure_optimizers()
        
        return sr_optimizers + vq_optimizers, sr_schedulers + vq_schedulers

    def save_sr_model_weights(self):
        # Save SR model weights when validation PSNR improves
        sr_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'sr_best_model.ckpt')
        self.sr_model.save_weights(sr_ckpt_path)
        print(f"Saved SR model weights to {sr_ckpt_path}")

    def save_vq_model_weights(self):
        # Save VQ model weights when validation reconstruction loss improves
        vq_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'vq_best_model.ckpt')
        self.vq_model.save_weights(vq_ckpt_path)
        print(f"Saved VQ model weights to {vq_ckpt_path}")
        
    def save_sr_epoch_end_weights(self):
        
        sr_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'sr_epoch_end_model.ckpt')
        self.sr_model.save_weights(sr_ckpt_path)
        print(f"Saved SR model weights at epoch end to {sr_ckpt_path}")

    def save_vq_epoch_end_weights(self):
        
        vq_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'vq_epoch_end_model.ckpt')
        self.vq_model.save_weights(vq_ckpt_path)
        print(f"Saved VQ model weights at epoch end to {vq_ckpt_path}")

class srvqDataModule(pl.LightningDataModule):
    def __init__(self, sr_data_module: SRDataModule, vq_data_module: pl.LightningDataModule):
        super().__init__()
        self.sr_data_module = sr_data_module
        self.vq_data_module = vq_data_module

    def setup(self, stage=None):
        self.sr_data_module.setup(stage)
        self.vq_data_module.setup(stage)

    def train_dataloader(self):
        
        loaders = {
            0: self.sr_data_module.train_dataloader(),
            1: self.vq_data_module.train_dataloader()
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        loaders = {
            0: self.sr_data_module.val_dataloader(),
            1: self.vq_data_module.val_dataloader()
        }
        return CombinedLoader(loaders, mode="max_size_cycle")



if __name__ == "__main__":


    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    data_module = SRDataModule(
        train_root='/data1/yifan/Super-Resolution-Neural-Operator-main/data/train',
        val_root='/data1/yifan/Super-Resolution-Neural-Operator-main/data/val',
        batch_size=64,
        inp_size=64,
        scale_min=2,
        scale_max=4,
        augment=True
     )
     
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        shared_layers = SharedLayers().cuda()

        srmodel = SRNO(shared_layers, width=256, blocks=16)
        sr_model = SRModel(srmodel).cuda()
        model = instantiate_from_config(config.model)
        

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency_train": 750,
                    "batch_frequency_val": 1,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, check_val_every_n_epoch=5, max_epochs=100, resume_from_checkpoint=os.path.join("/data1/yifan/taming-transformers-master/logs/2024-12-03T21-51-15_c2fmn_13/checkpoints/last.ckpt"), **trainer_kwargs)
        # trainer = Trainer.from_argparse_args(trainer_opt, check_val_every_n_epoch=5, max_epochs=100, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        
        joint_data_module = srvqDataModule(data_module, data)

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        
        joint_model = srvq(sr_model, model)
        
        # print(model)

        # run
        if opt.train:
            try:
                
                trainer.fit(joint_model, joint_data_module)
                
                total = sum([param.nelement() for param in model.parameters()])
                print("Number of parameter: %.2fM" % (total/1e6))
                input = torch.randn(1, 3, 256, 256)
                #c = torch.randn(1, 3, 256, 256)
                #flops, params = profile(model, inputs=(input,c))
                flops, params = profile(model, inputs=(input,))
                print(flops)
                print(params)
                print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
                print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(joint_model, joint_data_module)
            
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)