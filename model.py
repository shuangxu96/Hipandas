# -*- coding: utf-8 -*-
"""
高光谱图像处理神经网络模型定义

本模块实现了用于高光谱图像去噪、超分辨率重建的深度学习模型，
包括GDN（低秩分解去噪网络）、GSRN（低秩分解超分增强网络）和PRN（全色图像预测网络），
并定义了基础卷积模块和互引导层等关键组件。

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import os
# 设置环境变量以避免CUDA库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn


class DownsampleConv(nn.Module):
    """下采样卷积层
    
    通过卷积操作实现空间维度的下采样，替代传统的池化层，
    在减少特征图尺寸的同时保持参数学习能力。
    
    Args:
        in_c (int): 输入特征通道数
        out_c (int): 输出特征通道数
        kernel_size (int, optional): 卷积核大小. 默认值为2
        stride (int, optional): 卷积步长. 默认值为2
        padding (int, optional): 填充大小. 默认值为0
        bias (bool, optional): 是否使用偏置项. 默认值为True
    """
    def __init__(self, in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True):
        super(DownsampleConv, self).__init__()
        # 定义下采样卷积操作
        self.conv = nn.Conv2d(
            in_c, out_c, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征图，形状为[batch, in_c, height, width]
        
        Returns:
            torch.Tensor: 下采样后的特征图，形状为[batch, out_c, height/stride, width/stride]
        """
        return self.conv(x)


def conv2d(in_c, out_c, ks, stride=1, padding=None, dilation=1, groups=1, 
           bias=True, act=None, bn=False):
    """构建卷积模块
    
    生成包含卷积、可选批归一化和激活函数的顺序模块，
    简化网络层的构建过程。
    
    Args:
        in_c (int): 输入通道数
        out_c (int): 输出通道数
        ks (int): 卷积核大小
        stride (int, optional): 卷积步长. 默认值为1
        padding (int, optional): 填充大小. 若为None则自动计算为ks//2
        dilation (int, optional): 膨胀率. 默认值为1
        groups (int, optional): 分组卷积数量. 默认值为1
        bias (bool, optional): 是否使用偏置项. 默认值为True
        act (nn.Module, optional): 激活函数. 默认值为None
        bn (bool, optional): 是否使用批归一化. 默认值为False
    
    Returns:
        nn.Sequential: 包含卷积操作的顺序模块
    """
    # 自动计算填充大小，保持特征图尺寸不变
    if padding is None:
        padding = ks // 2
    
    # 构建基础卷积层
    layers = [
        nn.Conv2d(
            in_c, out_c, 
            kernel_size=ks, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
    ]
    
    # 添加批归一化层（若需要）
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    
    # 添加激活函数（若需要）
    if act is not None:
        layers.append(act)
    
    return nn.Sequential(*layers)


class MutualGuidanceLayer(nn.Module):
    """互引导层
    
    实现高光谱图像特征与全色图像特征之间的双向引导学习，
    通过跨模态特征融合增强特征表达能力。
    
    Args:
        n_feat (int): 特征通道数
        bias (bool): 是否使用偏置项
        act (nn.Module): 激活函数
        bn (bool, optional): 是否使用批归一化. 默认值为True
    """
    def __init__(self, n_feat, bias, act, bn=True):
        super().__init__()
        # 高光谱特征引导模块：输入为高光谱特征+全色特征的拼接
        self.head_hsi = conv2d(2 * n_feat, n_feat, 3, act=act, bn=bn, bias=bias)
        # 全色特征引导模块：输入为全色特征+高光谱特征的拼接
        self.head_pan = conv2d(2 * n_feat, n_feat, 3, act=act, bn=bn, bias=bias)

    def forward(self, hsi, pan):
        """前向传播
        
        Args:
            hsi (torch.Tensor): 高光谱特征图，形状为[batch, n_feat, h, w]
            pan (torch.Tensor): 全色特征图，形状为[batch, n_feat, h, w]
        
        Returns:
            tuple: 增强后的高光谱特征和全色特征
        """
        # 高光谱特征引导：拼接后通过卷积处理
        x = self.head_hsi(torch.cat([hsi, pan], dim=1))
        # 全色特征引导：拼接后通过卷积处理
        y = self.head_pan(torch.cat([pan, hsi], dim=1))
        return x, y


class GDN(nn.Module):
    """低秩分解去噪网络 (Guided Denoising Network)
    
    基于低秩矩阵分解理论的高光谱图像去噪网络，通过学习低秩表示实现噪声去除。
    
    Args:
        rank (int): 低秩分解的秩
        hsi_c (int): 高光谱图像的波段数（通道数）
        pan_c (int): 全色图像的通道数（通常为1）
        n_feat (int): 特征提取的通道数
        layerU (int, optional): U矩阵提取网络的层数. 默认值为5
        act (nn.Module, optional): 激活函数. 默认值为nn.LeakyReLU(1e-3)
        bias (bool, optional): 是否使用偏置项. 默认值为True
        sigmoid_V (bool, optional): 是否对V矩阵使用sigmoid激活. 默认值为True
        softmax_U (bool, optional): 是否对U矩阵使用softmax激活. 默认值为True
    """
    def __init__(self, 
                 rank, 
                 hsi_c,
                 pan_c,
                 n_feat, 
                 layerU=5, 
                 act=nn.LeakyReLU(1e-3), 
                 bias=True,
                 sigmoid_V=True,
                 softmax_U=True):
        super().__init__()
        
        self.device = 'cuda'  # 使用GPU加速
        self.sigmoid_V = sigmoid_V  # V矩阵是否使用sigmoid
        self.softmax_U = softmax_U  # U矩阵是否使用softmax
        self.aap = nn.AdaptiveAvgPool1d(rank)  # 自适应平均池化到指定秩
        
        # 构建U矩阵提取网络（用于学习低秩基）
        NetU = [
            conv2d(
                hsi_c, n_feat, 4, 
                stride=2, padding=1, 
                bias=bias, bn=True, act=act
            )
        ]
        # 添加中间下采样卷积层
        for i in range(layerU - 2):
            NetU.append(
                conv2d(
                    hsi_c, n_feat, 2, 
                    stride=2, padding=0, 
                    bias=bias, bn=True, act=act
                )
            )
        # 添加输出层，将特征映射回输入通道数
        NetU.append(
            conv2d(
                n_feat, hsi_c, 2, 
                stride=2, padding=0, 
                bias=bias, bn=True, act=act
            )
        )
        self.NetU = nn.Sequential(*NetU)

        # 构建V矩阵提取网络的头部（特征提取）
        self.NetV_head_hsi = conv2d(
            hsi_c, n_feat, 3, 
            bias=bias, bn=True, act=act
        )
        self.NetV_head_pan = conv2d(
            pan_c, n_feat, 3, 
            bias=bias, bn=True, act=act
        )
        
        # 构建V矩阵提取网络的主体（互引导模块）
        self.NetV_main = nn.ModuleList([
            MutualGuidanceLayer(n_feat, bias, act, True) for _ in range(5)
        ])
        
        # V矩阵提取网络的尾部（输出低秩系数）
        self.NetV_tail = nn.Conv2d(n_feat, rank, 3, padding=1, bias=bias)
    
    def getU(self, x):
        """提取低秩基矩阵U
        
        Args:
            x (torch.Tensor): 输入高光谱图像，形状为[batch, hsi_c, height, width]
        
        Returns:
            torch.Tensor: 低秩基矩阵U，形状为[batch, hsi_c, rank]
        """
        _, DD, HH, WW = x.shape
        # 通过下采样网络提取特征
        U = self.NetU(x)         # 形状为[batch, hsi_c, H/8, W/8]
        U = U.reshape(1, DD, -1)  # 展平空间维度：[batch, hsi_c, (H/8)*(W/8)]
        U = self.aap(U)          # 自适应池化到指定秩：[batch, hsi_c, rank]
        
        # 应用softmax归一化（若需要）
        if self.softmax_U:
            U = torch.softmax(U, dim=-1)
        return U
    
    def getV(self, hsi, pan):
        """提取低秩系数矩阵V
        
        Args:
            hsi (torch.Tensor): 输入高光谱图像，形状为[batch, hsi_c, height, width]
            pan (torch.Tensor): 输入全色图像，形状为[batch, pan_c, height, width]
        
        Returns:
            torch.Tensor: 低秩系数矩阵V，形状为[batch, rank, height*width]
        """
        _, DD, HH, WW = hsi.shape
        
        # 特征提取
        hsi_feat = self.NetV_head_hsi(hsi)
        pan_feat = self.NetV_head_pan(pan)
        
        # 多轮互引导学习
        for i in range(len(self.NetV_main)):
            hsi_feat, pan_feat = self.NetV_main[i](hsi_feat, pan_feat)
        
        # 生成V矩阵
        V = self.NetV_tail(hsi_feat)
        # 应用sigmoid激活（若需要）
        if self.sigmoid_V:
            V = V.sigmoid()
        # 展平空间维度
        V = V.reshape(1, -1, HH * WW)
        return V
    
    def forward(self, hsi, pan):
        """前向传播
        
        Args:
            hsi (torch.Tensor): 输入高光谱图像，形状为[batch, hsi_c, height, width]
            pan (torch.Tensor): 输入全色图像，形状为[batch, pan_c, height, width]
        
        Returns:
            tuple: 
                - 去噪后的高光谱图像，形状为[batch, hsi_c, height, width]
                - 低秩基矩阵U
                - 低秩系数矩阵V
        """
        _, DD, HH, WW = hsi.shape
        
        # 提取低秩分量
        U = self.getU(hsi)
        V = self.getV(hsi, pan)
        
        # 低秩重建：U和V的矩阵乘法
        y = torch.bmm(U, V)
        # 重塑为图像形状
        y = y.reshape(1, DD, HH, WW)
        
        return y, U, V


class GSRN(nn.Module):
    """低秩分解去噪增强网络 (Guided Super-resolution Network)
    
    在GDN基础上增加残差连接，用于高光谱图像超分辨率重建，
    结合全色图像的空间信息提升重建质量。
    
    注：参数含义与GDN相同，主要差异在于批归一化的使用和输出处理
    """
    def __init__(self, 
                 rank, 
                 hsi_c,
                 pan_c,
                 n_feat, 
                 layerU=5, 
                 act=nn.LeakyReLU(1e-3), 
                 bias=True,
                 sigmoid_V=True,
                 softmax_U=True):
        super().__init__()
        
        self.device = 'cuda'
        self.sigmoid_V = sigmoid_V
        self.softmax_U = softmax_U
        self.aap = nn.AdaptiveAvgPool1d(rank)
        
        # 构建U矩阵提取网络（不使用批归一化）
        NetU = [
            conv2d(
                hsi_c, n_feat, 4, 
                stride=2, padding=1, 
                bias=bias, bn=False, act=act
            )
        ]
        for i in range(layerU - 2):
            NetU.append(
                conv2d(
                    hsi_c, n_feat, 2, 
                    stride=2, padding=0, 
                    bias=bias, bn=False, act=act
                )
            )
        NetU.append(
            conv2d(
                n_feat, hsi_c, 2, 
                stride=2, padding=0, 
                bias=bias, bn=False, act=act
            )
        )
        self.NetU = nn.Sequential(*NetU)

        # 构建V矩阵提取网络（不使用批归一化）
        self.NetV_head_hsi = conv2d(
            hsi_c, n_feat, 3, 
            bias=bias, bn=False, act=act
        )
        self.NetV_head_pan = conv2d(
            pan_c, n_feat, 3, 
            bias=bias, bn=False, act=act
        )
        self.NetV_main = nn.ModuleList([
            MutualGuidanceLayer(n_feat, bias, act, False) for _ in range(5)
        ])
        self.NetV_tail = nn.Conv2d(n_feat, rank, 3, padding=1, bias=bias)

    def getU(self, x):
        """提取低秩基矩阵U（与GDN相同）"""
        _, DD, HH, WW = x.shape
        U = self.NetU(x)         # 形状为[batch, hsi_c, H/8, W/8]
        U = U.reshape(1, DD, -1)  # 展平空间维度
        U = self.aap(U)          # 自适应池化到指定秩
        if self.softmax_U:
            U = torch.softmax(U, dim=-1)
        return U
    
    def getV(self, hsi, pan):
        """提取低秩系数矩阵V（激活函数处理不同）"""
        _, DD, HH, WW = hsi.shape
        hsi_feat = self.NetV_head_hsi(hsi)
        pan_feat = self.NetV_head_pan(pan)
        
        for i in range(len(self.NetV_main)):
            hsi_feat, pan_feat = self.NetV_main[i](hsi_feat, pan_feat)
        
        V = self.NetV_tail(hsi_feat)
        if self.sigmoid_V:
            # 将输出范围调整到[-1, 1]
            V = 2 * (V.sigmoid() - 0.5)
        V = V.reshape(1, -1, HH * WW)
        return V
    
    def forward(self, hsi, pan, mode='test'):
        """前向传播（增加残差连接）
        
        Args:
            hsi (torch.Tensor): 输入高光谱图像
            pan (torch.Tensor): 输入全色图像
            mode (str, optional): 模式，'test'或'train'. 默认值为'test'
        
        Returns:
            tuple: 
                - 重建后的高光谱图像（含残差连接）
                - 低秩基矩阵U
                - 低秩系数矩阵V
        """
        _, DD, _, _ = hsi.shape
        _, _, HH, WW = pan.shape
        
        # 提取低秩分量
        U = self.getU(hsi)
        V = self.getV(hsi, pan)

        # 低秩重建并添加残差连接
        y = torch.bmm(U, V)
        y = y.reshape(1, DD, HH, WW) + hsi  # 残差连接：学习残差而非完整图像

        return y, U, V


class PRN(nn.Module):
    """全色图像预测网络 (Panchromatic Reconstruction Network)
    
    从高光谱图像预测对应的全色图像，用于训练中的一致性约束。
    
    Args:
        hsi_c (int): 高光谱图像的波段数
        pan_c (int): 全色图像的通道数（通常为1）
        n_feat (int): 特征提取的通道数
        layer (int, optional): 网络层数. 默认值为5
        act (nn.Module, optional): 激活函数. 默认值为nn.LeakyReLU(1e-3)
        bias (bool, optional): 是否使用偏置项. 默认值为True
    """
    def __init__(self, 
                 hsi_c,
                 pan_c,
                 n_feat, 
                 layer=5, 
                 act=nn.LeakyReLU(1e-3), 
                 bias=True):
        super().__init__()

        # 构建主干网络
        main = [conv2d(hsi_c, n_feat, 3, bias=bias, act=act)]
        # 添加中间卷积层
        for i in range(layer - 2):
            main.append(conv2d(n_feat, n_feat, 3, bias=bias, act=act))
        # 输出层：将特征映射到全色图像通道
        main.append(nn.Conv2d(n_feat, pan_c, 3, padding=1, bias=bias))
        
        self.main = nn.Sequential(*main)

    def forward(self, hsi):
        """前向传播
        
        Args:
            hsi (torch.Tensor): 输入高光谱图像，形状为[batch, hsi_c, height, width]
        
        Returns:
            torch.Tensor: 预测的全色图像，形状为[batch, pan_c, height, width]
        """
        return self.main(hsi)