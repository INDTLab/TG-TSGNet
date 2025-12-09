import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models.galerkin import simple_attn
from models import register
from utils import make_coord
from utils import show_feature_map
from tcm_20 import Encoder

@register('sronet')
class SRNO(nn.Module):

    def __init__(self, encoder_spec, width=256, blocks=16):
        super().__init__()
        self.width = width
        
        # self.encoder = models.make(encoder_spec)    # edsr-baseline
        self.encoder = Encoder(
            ch=64,
            out_ch=3,
            num_res_blocks=2,
            attn_resolutions=[16],
            in_channels=3,
            resolution=256,
            z_channels=256
        )
        # 1x1 卷积，用于将输入的多通道数据转换为指定的宽度
        # self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)
        # self.conv00 = nn.Conv2d(2058, self.width, 1)
        self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)

        # 自定义的简单注意力层
        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        #self.conv2 = simple_attn(self.width, blocks)
        #self.conv3 = simple_attn(self.width, blocks)

        # 用于输出层的全连接卷积层，用于将高维特征图逐步映射到 3 通道的 RGB 输出
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)

    # 生成特征
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    # 输入图像坐标 coord 和网格单元尺寸 cell
    def query_rgb(self, coord, cell):      
        feat = (self.feat)  # 由 gen_feat 函数得到特征图
        grid = 0

        # 低分辨率特征图中的坐标网格
        # 创建一个和特征图尺寸一致的坐标网格，用于后续的坐标变换和插值操作
        # pos_lr 的形状为 [batch_size, 2, height, width]
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # 生成相对坐标和特征
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        # 将相对坐标、特征图和相对单元格信息拼接在一起，形成一个包含空间、特征和尺度信息的综合张量 grid
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        
        # 原图插值可以大致给出结构，但细节会模糊。于是我们把网络的输出（细节）和插值图加起来，得到高质量的图像。
        # ret（网络输出） = 网络预测的高频细节 + 插值得到的粗略图（来自原图）
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
