import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from taming.modules.diffusionmodules.vqhr import NCB, NTB, Downsample, Upsample
from models.galerkin import simple_attn
from models import register
from utils import make_coord
from utils import show_feature_map



class SRNO(nn.Module):
    def __init__(self, shared_layers, width=256, blocks=16):
        super().__init__()
        self.width = width
        
        self.shared_layers = shared_layers
        self.up1 = Upsample(64*2, 64, with_conv=True)
        
        
        self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)

        # 自定义的简单注意力层
        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)

        # 用于输出层的全连接卷积层，用于将高维特征图逐步映射到 3 通道的 RGB 输出
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)

    # 生成特征
    def gen_feat(self, inp):
    
        self.inp = inp
        self.feat = self.shared_layers(inp)
        self.feat = self.up1(self.feat)
        # print(f"feat requires_grad: {self.feat.requires_grad}")
        
        return self.feat


    # 输入图像坐标 coord 和网格单元尺寸 cell
    def query_rgb(self, coord, cell):  
    
        feat = (self.feat)
        grid = 0

        # 低分辨率特征图中的坐标网格，它创建一个和特征图尺寸一致的坐标网格，用于后续的坐标变换和插值操作
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
         
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)
        
        # print("Grid shape:", grid.shape)
        
        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        

        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret
    
    
    def forward(self, inp, coord, cell):
    
        # print_grad_status(self)
        
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
        

    # 使用编码器提取特征，结合相对坐标与注意力机制进行插值，并最终生成高分辨率的 RGB 图像输出



class SharedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.shared_block_1 = NCB(64, 64, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        self.shared_block_2 = NCB(64, 64, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        
        self.down1 = Downsample(64,64*2, with_conv=True)
        
        self.shared_block_3 = NTB(64*2, 64*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.shared_block_4 = NTB(64*2, 64*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv_in(x)
        x = self.shared_block_1(x)
        x = self.shared_block_2(x)
        x = self.down1(x)
        x = self.shared_block_3(x)
        x = self.shared_block_4(x)
        
        return x

        
        