"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .hybrid_encoder import HybridEncoder

from functools import partial
from ..extre_module.custom_nn.module.FiGCA import FiGCA_EncoderBlock
from ..extre_module.ultralytics_nn.block import C2f_Block
from ..extre_module.custom_nn.downsample.SHWD import LAWD

from ..core import register

__all__ = ['HybridEncoder_C2f_FiGCA']


@register()
class HybridEncoder_C2f_FiGCA(HybridEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],  # 输入特征图的通道数列表，例如来自骨干网络的不同层
                 feat_strides=[8, 16, 32],  # 输入特征图的步幅列表，表示特征图相对于输入图像的缩放比例
                 hidden_dim=256,  # 隐藏层维度，所有特征图将被投影到这个维度
                 nhead=8,  # Transformer 编码器中多头自注意力的头数
                 dim_feedforward=1024,  # Transformer 编码器中前馈网络的维度
                 dropout=0.0,  # Transformer 编码器中的 dropout 概率
                 enc_act='gelu',  # Transformer 编码器中的激活函数类型
                 use_encoder_idx=[2],  # 指定哪些层使用 Transformer 编码器（索引列表）
                 num_encoder_layers=1,  # Transformer 编码器的层数
                 pe_temperature=10000,  # 位置编码的温度参数，用于控制频率
                 expansion=1.0,  # FPN 和 PAN 中特征扩展因子
                 depth_mult=1.0,  # 深度乘数，用于调整网络深度
                 act='silu',  # FPN 和 PAN 中使用的激活函数类型
                 eval_spatial_size=None,  # 评估时的空间尺寸 (H, W)，用于预计算位置编码
                 version='dfine',  # 模型版本，决定使用哪些具体模块（如 'dfine' 或其他）
                 ):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout,
                         enc_act, use_encoder_idx, num_encoder_layers, pe_temperature,
                         expansion, depth_mult, act, eval_spatial_size, version)
        self.fpn_blocks = nn.ModuleList()  # FPN 融合块
        for _ in range(len(in_channels) - 1, 0, -1):  # 从高层到低层遍历
            self.fpn_blocks.append(
                C2f_Block(hidden_dim * 2, hidden_dim, partial(FiGCA_EncoderBlock), n=round(3 * depth_mult))
            )

        self.pan_blocks = nn.ModuleList()  # PAN 融合块
        for _ in range(len(in_channels) - 1):  # 从低层到高层遍历
            # 下采样卷积，将低层特征下采样以匹配高层特征尺寸
            self.downsample_convs.append(
                nn.Sequential(LAWD(hidden_dim, hidden_dim,)) \
            )
            # PAN 块，融合下采样后的低层特征和高层特征
            self.pan_blocks.append(
                C2f_Block(hidden_dim * 2, hidden_dim, partial(FiGCA_EncoderBlock), n=round(3 * depth_mult))
            )
