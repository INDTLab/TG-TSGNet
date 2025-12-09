import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from skimage.metrics import structural_similarity as ssim



class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        img1_np = img1.detach().permute(0, 2, 3, 1).cpu().numpy()  
        img2_np = img2.detach().permute(0, 2, 3, 1).cpu().numpy()  
        
        ssim_value = 0
        for i in range(img1_np.shape[0]):
            data_range = img2_np[i].max() - img2_np[i].min()
            ssim_value += ssim(img1_np[i], img2_np[i], multichannel=True, channel_axis=-1, data_range=data_range)
        
        return 1 - (torch.tensor(ssim_value / img1_np.shape[0]))

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, ssim_weight = 1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        # LPIPS利用预训练的神经网络对图像进行特征提取，比较输入图像和重建图像在高层特征空间中的差异。这种差异通常能更好地反映人眼感知的图像相似度
        # 感知损失更加关注图像的感知质量，即图像在视觉上的相似性。它有助于保留更多的纹理细节和视觉结构
        self.perceptual_loss = LPIPS().eval()

        self.perceptual_weight = perceptual_weight

        self.ssim_loss = SSIMLoss()  # 实例化SSIM损失类
        self.ssim_weight = ssim_weight  # 保存SSIM损失权重

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        # 重建损失 L1损失，也叫做平均绝对误差（MAE），它计算输入图像与重建图像之间的像素差异的绝对值
        # L1损失注重整体像素的准确性，即图像中每个像素值之间的差异。它有助于生成看起来平滑、精确的重建图像，但对细节和感知质量的提升有限
        # print(f"inputs shape: {inputs.shape}")
        # print(f"reconstructions shape: {reconstructions.shape}")
        
        
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        ssim_loss_value = self.ssim_loss(inputs.contiguous(), reconstructions.contiguous())

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)
        rec_loss = rec_loss + self.ssim_weight * ssim_loss_value

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),     # 总loss
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),    # 量化损失
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),           # 负对数似然损失
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),           # 重建损失 rec
                   "{}/p_loss".format(split): p_loss.detach().mean(),               # 感知损失 Perceptual Loss
                   "{}/ssim_loss".format(split): ssim_loss_value.detach().mean(),   # 记录SSIM损失
                   "{}/d_weight".format(split): d_weight.detach(),                  # 判别器权重
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),       # 判别器因子
                   "{}/g_loss".format(split): g_loss.detach().mean(),               # 生成器损失
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
