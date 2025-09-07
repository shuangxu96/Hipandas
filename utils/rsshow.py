# -*- coding: utf-8 -*-
"""
遥感图像可视化工具函数模块

该模块提供了一系列一系列用于遥感图像（包括高光谱图像HSI、多光谱图像MSI和全色图像PAN）可视化的工具函数，
支持NumPy数组和PyTorch张量两种数据类型的输入，并提供线性拉伸等增强处理以优化视图效果。

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import torch
import numpy as np
from matplotlib.pyplot import imshow  # 用于图像显示

def im2double(I):
    """
    将图像数据转换为双精度浮点数格式（归一化到[0, 1]范围）
    
    该函数处理不同数据类型（uint8、uint16）的输入图像，将其归一化到[0, 1]区间的浮点数，
    支持NumPy数组和PyTorch张量两种输入类型。
    
    参数:
        I (np.ndarray 或 torch.Tensor): 输入图像，可以是二维（单波段）或三维（多波段）
            - 若为uint8类型，除以255进行归一化
            - 若为uint16类型，除以65535进行归一化
    
    返回:
        np.ndarray 或 torch.Tensor: 归一化后的图像，保持与输入相同的数据类型
    """
    if isinstance(I, np.ndarray):
        if I.dtype == np.uint8:
            I = I.astype(np.float32) / 255.
        elif I.dtype == np.uint16:
            I = I.astype(np.float32) / 65535.
    elif isinstance(I, torch.Tensor):
        if I.dtype == torch.uint8:
            I = I.float() / 255.
        elif I.dtype == torch.uint16:
            I = I.float() / 65535.
    return I

def hsi2tensor(hsi):
    """
    将高光谱图像（HSI）从NumPy数组转换为PyTorch张量
    
    转换过程包括维度重排，将原始的(H, W, C)格式转换为PyTorch常用的(1, C, H, W)格式，
    其中1表示批次维度，C表示光谱波段数，H和W表示空间维度。
    
    参数:
        hsi (np.ndarray): 输入高光谱图像，形状为(H, W, C)
    
    返回:
        torch.Tensor: 转换后的张量，形状为(1, C, H, W)
    """
    return torch.tensor(hsi).permute(2, 0, 1)[None, ...]

def tensor2hsi(hsi):
    """
    将高光谱图像（HSI）从PyTorch张量转换为NumPy数组
    
    转换过程包括移除批次维度、维度重排和数据迁移（从GPU到CPU），
    将PyTorch常用的(1, C, H, W)格式转换为原始的(H, W, C)格式。
    
    参数:
        hsi (torch.Tensor): 输入张量，形状为(1, C, H, W)
    
    返回:
        np.ndarray: 转换后的数组，形状为(H, W, C)
    """
    return np.array(hsi.detach().squeeze().cpu().permute(1, 2, 0))

def tensor2hpan(hsi):
    """
    将全色图像（PAN）从PyTorch张量转换为NumPy数组
    
    适用于单波段的全色图像，移除批次和通道维度，并将数据从GPU迁移到CPU。
    
    参数:
        hsi (torch.Tensor): 输入张量，形状为(1, 1, H, W)
    
    返回:
        np.ndarray: 转换后的数组，形状为(H, W)
    """
    return np.array(hsi.detach().squeeze().cpu())

def pan2tensor(pan):
    """
    将全色图像（PAN）从NumPy数组转换为PyTorch张量
    
    为单波段全色图像添加批次和通道维度，转换为PyTorch常用的(1, 1, H, W)格式。
    
    参数:
        pan (np.ndarray): 输入全色图像，形状为(H, W)
    
    返回:
        torch.Tensor: 转换后的张量，形状为(1, 1, H, W)
    """
    return torch.tensor(pan[None, None, ...])

def linear_stretch(I, scale):
    """
    对图像进行线性拉伸以增强对比度
    
    通过去除图像中极端亮度值（使用分位数计算），将图像像素值线性映射到[0, 1]范围，
    有效增强图像的视觉对比度，尤其适用于动态范围较广的遥感图像。
    
    参数:
        I (np.ndarray): 输入图像，二维（单波段）或三维（多波段）
        scale (float): 拉伸比例，用于计算分位数的阈值（0 < scale < 0.5）
    
    返回:
        np.ndarray: 拉伸后的图像，像素值范围为[0, 1]
    """
    # 计算上下分位数阈值
    q = np.quantile(I.flatten(), [scale, 1 - scale])
    low, high = q
    # 截断极端值
    I[I > high] = high
    I[I < low] = low
    # 线性映射到[0, 1]
    I = (I - low) / (high - low)
    return I

def rsshow_np(I, scale=0.01, band=None, ignore_value=None):
    """
    显示NumPy数组格式的遥感图像并进行可视化增强
    
    该函数是处理NumPy数组的核心可视化函数，支持高光谱/多光谱图像（HSI/MSI）和全色图像（PAN），
    提供波段选择、线性拉伸和异常值处理功能，最终将图像转换为uint8格式用于显示。
    
    参数:
        I (np.ndarray): 输入图像
            - 高光谱/多光谱图像：形状为(H, W, C)，C为波段数
            - 全色图像：形状为(H, W)
        scale (float, 可选): 线性拉伸的比例，默认0.01（即去除1%的极端值）
        band (list或None, 可选): 选择显示的波段索引，None时自动选择3个代表性波段
        ignore_value (float或None, 可选): 需要忽略的异常值，设置为NaN以便显示时透明
    
    返回:
        np.ndarray: 处理后用于显示的图像，uint8格式，形状为(H, W, 3)或(H, W)
    """
    # 转换为[0, 1]范围的浮点数
    I = im2double(I)
    
    # 波段选择：若为多波段图像且未指定波段，自动选择3个代表性波段
    if band is None:
        if I.ndim == 3 and I.shape[-1] > 3:
            C = I.shape[-1]
            # 选择近红外、中红外和可见光波段（根据常见高光谱数据分布）
            band = [C-1, int(C*0.5), int(C*0.1)]
            I = I[:, :, band]
    else:
        I = I[:, :, band]

    Iq = I
    # 处理负值（设置为0）
    Iq[Iq < 0] = 0
    # 处理需要忽略的值（设置为NaN）
    if ignore_value is not None:
        Iq[Iq == ignore_value] = np.nan

    # 对单波段图像直接拉伸，多波段图像逐波段拉伸
    if I.ndim == 2:
        I = linear_stretch(I, scale)
    else:
        for i in range(3):
            I[..., i] = linear_stretch(I[..., i], scale)
    
    # 转换为uint8格式（0-255）并显示
    I = (255. * I).astype(np.uint8)
    imshow(I)
    return I

def rsshow_torch(I, scale=0.01, band=None, ignore_value=None):
    """
    显示PyTorch张量格式的遥感图像并进行可视化增强
    
    该函数是处理PyTorch张量的可视化入口，首先将张量转换为NumPy数组，
    然后调用rsshow_np函数进行后续处理和显示。
    
    参数:
        I (torch.Tensor): 输入张量
            - 高光谱/多光谱图像：形状为(1, C, H, W)
            - 全色图像：形状为(1, 1, H, W)
        scale (float, 可选): 线性拉伸的比例，默认0.01
        band (list或None, 可选): 选择显示的波段索引
        ignore_value (float或None, 可选): 需要忽略的异常值
    
    返回:
        np.ndarray: 处理后用于显示的图像，uint8格式
    """
    # 将张量转换为NumPy数组，调整维度为(H, W, C)或(H, W)
    I = np.array(I.detach().permute(0, 2, 3, 1).squeeze(0).cpu())
    if I.shape[-1] == 1:
        I = I[..., 0]  # 移除单波段维度
    # 调用NumPy版本的显示函数
    I = rsshow_np(I, scale=scale, band=band, ignore_value=ignore_value)
    return I

def rsshow(I, scale=0.01, band=None, ignore_value=None):
    """
    遥感图像可视化的统一接口函数
    
    根据输入数据类型（NumPy数组或PyTorch张量）自动选择对应的处理函数，
    提供一致的接口用于显示各种类型的遥感图像。
    
    参数:
        I (np.ndarray 或 torch.Tensor): 输入图像，可以是NumPy数组或PyTorch张量
        scale (float, 可选): 线性拉伸的比例，默认0.01
        band (list或None, 可选): 选择显示的波段索引
        ignore_value (float或None, 可选): 需要忽略的异常值
    
    返回:
        np.ndarray: 处理后用于显示的图像，uint8格式
    """
    if isinstance(I, np.ndarray):
        I = rsshow_np(I, scale=scale, band=band, ignore_value=ignore_value)
    elif isinstance(I, torch.Tensor):
        I = rsshow_torch(I, scale=scale, band=band, ignore_value=ignore_value)
    return I