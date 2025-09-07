# -*- coding: utf-8 -*-
"""
通用工具函数集合
功能：包含数据格式转换、图像增强、随机种子设置、图像裁剪、缩放等基础操作

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import numpy as np
import torch
import random
from skimage.io import imshow, imread, imsave  # 图像显示与读写
import torch.nn.functional as F  # PyTorch函数式接口
import utils  # 导入其他工具模块


def setup_seed(seed):
    """
    设置随机种子，保证实验可复现性
    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
    np.random.seed(seed)  # 设置NumPy随机种子
    random.seed(seed)  # 设置Python随机种子


def tensor2array(x, clip=True):
    """
    将PyTorch张量转换为NumPy数组
    参数:
        x: 输入张量，可为4维图像[1, C, H, W]或3维信号[1, C, N]
        clip: 是否将值裁剪到[0, 1]范围
    返回:
        np.array: 转换后的数组，图像为[H, W, C]，信号为[N, C]
    """
    if x.ndim == 4:  # 图像张量[1, C, H, W]
        x = np.array(x.detach().squeeze(0).permute(1, 2, 0).cpu())  # 移除批量维度并转置为[H, W, C]
    elif x.ndim == 3:  # 信号张量[1, C, N]
        x = np.array(x.detach().squeeze(0).permute(1, 0).cpu())  # 转置为[N, C]
    if clip:
        x = np.clip(x, 0., 1.)  # 裁剪到[0, 1]
    return x


def array2tensor(x, device):
    """
    将NumPy数组转换为PyTorch张量
    参数:
        x: 输入数组，形状为[H, W, C]
        device: 目标设备（如'cuda'或'cpu'）
    返回:
        torch.Tensor: 转换后的张量，形状为[1, C, H, W]
    """
    x = torch.tensor(x, dtype=torch.float32, device=device)  # 转换为张量并移动到目标设备
    x = x.permute(2, 0, 1)[None, ...]  # 转置为[C, H, W]并添加批量维度
    return x


def center_crop(img, win_size):
    """
    对图像进行中心裁剪
    参数:
        img: 输入图像，可为3维[H, W, C]或2维[H, W]
        win_size: 裁剪窗口大小（正方形）
    返回:
        裁剪后的图像，形状为[win_size, win_size, C]或[win_size, win_size]
    """
    if img.ndim == 3:
        H, W, _ = img.shape
    elif img.ndim == 2:
        H, W = img.shape
    h = H // 2  # 中心行坐标
    w = W // 2  # 中心列坐标
    # 计算裁剪区域
    x1 = h - win_size // 2
    x2 = h + win_size // 2
    y1 = w - win_size // 2
    y2 = w + win_size // 2
    return img[x1:x2, y1:y2, ...]


def data_aug(image, mode=None):
    """
    对图像进行数据增强（用于训练，提高模型泛化能力）
    参数:
        image: 输入张量，形状[N, C, H, W]
        mode: 增强模式（0-7，None则随机选择）
    返回:
        image: 增强后的图像
        mode: 使用的增强模式（用于逆操作）
    """
    dims = (-1, -2)  # 空间维度（高度和宽度）

    if mode is None:
        mode = random.randint(0, 7)  # 随机选择增强模式
    if mode == 0:
        # 原始图像（无增强）
        image = image
    elif mode == 1:
        # 上下翻转
        image = torch.flipud(image)
    elif mode == 2:
        # 逆时针旋转90度
        image = torch.rot90(image, dims=dims)
    elif mode == 3:
        # 旋转90度后上下翻转
        image = torch.rot90(image, dims=dims)
        image = torch.flipud(image)
    elif mode == 4:
        # 旋转180度
        image = torch.rot90(image, k=2, dims=dims)
    elif mode == 5:
        # 旋转180度后上下翻转
        image = torch.rot90(image, k=2, dims=dims)
        image = torch.flipud(image)
    elif mode == 6:
        # 旋转270度
        image = torch.rot90(image, k=3, dims=dims)
    elif mode == 7:
        # 旋转270度后上下翻转
        image = torch.rot90(image, k=3, dims=dims)
        image = torch.flipud(image)

    return image, mode


def inv_data_aug(image, mode):
    """
    数据增强的逆操作（用于测试时恢复原始方向）
    参数:
        image: 增强后的图像张量[N, C, H, W]
        mode: 增强模式（与data_aug中使用的模式一致）
    返回:
        恢复原始方向的图像
    """
    dims = (-1, -2)

    if mode == 0:
        # 原始图像（无需操作）
        image = image
    elif mode == 1:
        # 上下翻转的逆操作（再次上下翻转）
        image = torch.flipud(image)
    elif mode == 2:
        # 旋转90度的逆操作（顺时针旋转90度）
        image = torch.rot90(image, k=-1, dims=dims)
    elif mode == 3:
        # 旋转90度+翻转的逆操作（先翻转再旋转）
        image = torch.flipud(image)
        image = torch.rot90(image, k=-1, dims=dims)
    elif mode == 4:
        # 旋转180度的逆操作（再次旋转180度）
        image = torch.rot90(image, k=-2, dims=dims)
    elif mode == 5:
        # 旋转180度+翻转的逆操作（先翻转再旋转）
        image = torch.flipud(image)
        image = torch.rot90(image, k=-2, dims=dims)
    elif mode == 6:
        # 旋转270度的逆操作（顺时针旋转270度）
        image = torch.rot90(image, k=-3, dims=dims)
    elif mode == 7:
        # 旋转270度+翻转的逆操作（先翻转再旋转）
        image = torch.flipud(image)
        image = torch.rot90(image, k=-3, dims=dims)
        
    return image


def show_img(img):
    """
    显示图像（自动处理张量或数组）
    参数:
        img: 输入图像（张量或NumPy数组）
    """
    if isinstance(img, torch.Tensor):
        img = tensor2array(img, clip=False)  # 张量转数组
    imshow(img)  # 显示图像


def im2double(x):
    """
    将uint8图像转换为[0,1]范围的float32
    参数:
        x: 输入图像（uint8）
    返回:
        转换后的图像（float32）
    """
    x = x.astype(np.float32)
    x = x / 255.
    return x


def imresize(img, ratio, mode='bicubic'):
    """
    图像缩放（基于PyTorch的插值函数）
    参数:
        img: 输入张量，形状[N, C, H, W]
        ratio: 缩放比例
        mode: 插值方式（'bicubic'、'bilinear'等）
    返回:
        缩放后的张量
    """
    return F.interpolate(img, scale_factor=ratio, mode=mode, align_corners=True)


def downsample_hsi(hsi, ratio):
    """
    高光谱图像下采样（先模糊再下采样，模拟光学低通滤波）
    参数:
        hsi: 输入高光谱张量[N, C, H, W]
        ratio: 下采样比例（如4表示1/4）
    返回:
        下采样后的张量
    """
    lp_hsi = utils.mtf(hsi, 'none', ratio, mode='replicate')  # 应用MTF低通滤波
    return imresize(lp_hsi, ratio=1/ratio)  # 下采样