import random
import math
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord

# 调整大小
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

# 快速下采样
@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset              # HR数据集
        self.inp_size = inp_size            # 哥们怎么能把input缩写成inp 我以为什么高级代名词呢 低分辨率图像的目标尺寸
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment              # 是否启用数据增强

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]                             # 获取对应的图像
        s = random.uniform(self.scale_min, self.scale_max)  # 在 scale_min 和 scale_max 之间随机选择一个缩放因子 s，用于生成低分辨率图像

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)     # 计算低分辨率图像的高度 h_lr，通过将高分辨率图像高度除以缩放因子 s 并向下取整
            w_lr = math.floor(img.shape[-1] / s + 1e-9)     # 计算低分辨率图像的宽度 w_lr，通过将高分辨率图像高度除以缩放因子 s 并向下取整
            # 计算高分辨率尺寸
            # 通过将 h_lr 乘以缩放因子 s 并四舍五入，得到对应的高分辨率高度 h_hr
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            # 裁剪出来
            img = img[:, :h_hr, :w_hr]
            # 生成低分辨率图像
            img_down = resize_fn(img, (h_lr, w_lr))
            # 低分辨率图像和高分辨率图像分别赋值给 crop_lr 和 crop_hr
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
        # 反正最后这个 低分辨率图像 crop_lr 和对应的高分辨率图像 crop_hr，用于训练模型
        # 是否进行数据增强 水平 垂直 同时
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # 将数据增强应用于低分辨率和高分辨率图像
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # 生成高分辨率坐标网格和高分辨率图像
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        # 随机采样高分辨率坐标和 RGB 值
        if self.inp_size is not None:

            # 从高分辨率图像的像素总数 h_hr * w_hr 中随机选择 h_lr * w_lr 个不重复的索引，存储在 idx 中
            # 这些索引用于从高分辨率图像中随机采样对应的坐标和 RGB 值
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])            # 将高分辨率坐标网格展平为二维张量，每行对应一个坐标点
            hr_coord = hr_coord[idx, :]                                 # 使用随机索引 idx 选择对应的坐标点
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])    # 将选择后的坐标点重新调整为 (h_lr, w_lr, coord_dim) 的形状
            # 从高分辨率图像中随机选择一部分像素点，用于训练模型。这种方法可以减小计算量，并促进模型学习局部细节

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)          # 将选择后的 RGB 值重新调整为 (channels, h_lr, w_lr) 的形状
            # 将高分辨率图像 crop_hr 的坐标和像素值调整为低分辨率图像的尺寸
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        # 计算单元格大小 cell 用于坐标归一化
        
        return {
            'inp': crop_lr,         # 低分辨率图像张量，用作模型的输入
            'coord': hr_coord,      # 高分辨率坐标网格张量，对应于高分辨率图像的坐标信息
            'cell': cell,           # 单元格大小张量，用于坐标归一化
            'gt': hr_rgb            # 高分辨率图像张量，作为模型的目标输出
        }    
