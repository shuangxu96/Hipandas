# -*- coding: utf-8 -*-
"""
高光谱图像噪声模型工具库

该模块实现了多种适用于高光谱图像（HSI）的噪声生成函数和数据增强工具，
支持高斯噪声（包括iid和non-iid模式）、脉冲噪声、条纹噪声、截止线噪声以及混合噪声的模拟。

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import torch
import torchvision
import random
import cv2

# 图像转换工具与数据集处理类
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomHorizontalFlip, RandomChoice
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset, TensorDataset, ResampleDataset

from PIL import Image
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter
import numpy as np


def istensor(img):
    """
    判断输入是否为PyTorch张量
    
    参数:
        img: 待检测对象
        
    返回:
        bool: 若为torch.Tensor则返回True，否则返回False
    """
    return isinstance(img, torch.Tensor)


def isarray(img):
    """
    判断输入是否为NumPy数组
    
    参数:
        img: 待检测对象
        
    返回:
        bool: 若为np.ndarray则返回True，否则返回False
    """
    return isinstance(img, np.ndarray)


def clip(img, minval, maxval):
    """
    对图像像素值进行截断处理
    
    将图像中所有超过maxval的像素值设为maxval，低于minval的像素值设为minval，
    用于确保添加噪声后的图像像素值在有效范围内。
    
    参数:
        img (np.ndarray或torch.Tensor): 输入图像
        minval (float): 最小值阈值
        maxval (float): 最大值阈值
        
    返回:
        截断处理后的图像，数据类型与输入一致
    """
    img[img > maxval] = maxval
    img[img < minval] = minval
    return img


class AddGauss(object):
    """
    为图像添加高斯噪声（iid模式）
    
    高斯噪声是图像复原任务中最常用的噪声类型之一，iid（独立同分布）模式表示
    所有像素和波段共享相同的噪声标准差，适用于模拟传感器电子噪声。
    
    属性:
        sigma_ratio (float): 噪声标准差与255的比值（将噪声强度从[0,255]映射到[0,1]范围）
    """

    def __init__(self, sigma):
        """
        初始化高斯噪声生成器
        
        参数:
            sigma (int): 噪声标准差（像素值范围为[0,255]时的参数）
        """
        self.sigma_ratio = sigma / 255.  # 转换为[0,1]范围下的标准差

    def __call__(self, img):
        """
        向输入图像添加高斯噪声
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)，其中B为波段数，H为高度，W为宽度
            
        返回:
            np.ndarray: 添加噪声后的图像，像素值范围被截断在[0,1]
        """
        # 生成与图像形状相同的高斯噪声
        noise = np.random.randn(*img.shape) * self.sigma_ratio
        # 添加噪声并截断像素值范围
        return clip(img + noise, 0., 1.)


class AddGaussBlind(object):
    """
    为图像添加盲高斯噪声
    
    盲高斯噪声的标准差在指定范围内随机选择，适用于模拟噪声强度未知的场景，
    可增强模型对不同噪声水平的鲁棒性。
    """

    def __init__(self, min_sigma, max_sigma):
        """
        初始化盲高斯噪声生成器
        
        参数:
            min_sigma (int): 噪声标准差最小值（[0,255]范围）
            max_sigma (int): 噪声标准差最大值（[0,255]范围）
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        """
        向输入图像添加随机强度的高斯噪声
        
        参数:
            img (np.ndarray或torch.Tensor): 输入图像
            
        返回:
            添加噪声后的图像，像素值范围被截断在[0,1]
        """
        # 在[min_sigma, max_sigma]范围内随机选择标准差
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        
        if isarray(img):
            # 为NumPy数组生成噪声
            noise = np.random.randn(*img.shape) * sigma
        elif istensor(img):
            # 为PyTorch张量生成噪声
            noise = torch.randn(*img.shape) * sigma
            
        return clip(img + noise, 0., 1.)


class AddGaussNoniid(object):
    """
    为图像添加非独立同分布（non-iid）高斯噪声
    
    高光谱图像的不同波段往往受到不同程度的噪声干扰，non-iid高斯噪声为每个波段
    分配独立的随机标准差，更接近真实遥感成像场景。
    """

    def __init__(self, min_sigma, max_sigma, channel_first=True):
        """
        初始化非iid高斯噪声生成器
        
        参数:
            min_sigma (int): 每个波段噪声标准差的最小值（[0,255]范围）
            max_sigma (int): 每个波段噪声标准差的最大值（[0,255]范围）
            channel_first (bool): 图像通道是否在第一维度（True表示(B,H,W)，False表示(H,W,B)）
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.channel_first = channel_first

    def __call__(self, img):
        """
        向输入图像的每个波段添加独立高斯噪声
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)
            
        返回:
            np.ndarray: 添加噪声后的图像，像素值范围被截断在[0,1]
        """
        out = img.copy()  # 复制原图以避免修改输入
        B, H, W = img.shape  # 获取波段数、高度和宽度
        
        # 为每个波段生成独立的噪声
        for i in range(B):
            # 为当前波段随机选择标准差
            sigma = np.random.uniform(self.min_sigma, self.max_sigma, 1) / 255
            # 生成与当前波段匹配的噪声
            noise = np.random.randn(1, H, W) * sigma
            out[i:i+1, ...] = out[i:i+1, ...] + noise  # 添加噪声到当前波段
            
        return clip(out, 0., 1.)


class AddNoiseMixed(object):
    """
    为图像添加混合噪声
    
    混合噪声模型可组合多种类型的噪声（如高斯噪声+脉冲噪声），并为不同波段分配
    不同类型的噪声，更接近真实高光谱图像的复杂噪声场景。
    """

    def __init__(self, noise_bank, num_bands):
        """
        初始化混合噪声生成器
        
        参数:
            noise_bank (list): 噪声生成器列表（如[_AddGaussNoniid, _AddNoiseImpulse]）
            num_bands (list): 每个噪声生成器作用的波段数量或比例
            
        异常:
            AssertionError: 若noise_bank与num_bands长度不匹配则触发
        """
        assert len(noise_bank) == len(num_bands), "噪声生成器与波段数量列表长度必须一致"
        self.noise_bank = noise_bank  # 噪声生成器集合
        self.num_bands = num_bands    # 每个噪声作用的波段数

    def __call__(self, img):
        """
        向图像添加混合噪声
        
        随机分配波段到不同噪声类型，然后应用对应的噪声生成器。
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)
            
        返回:
            np.ndarray: 添加混合噪声后的图像
        """
        out = img.copy()
        B, H, W = img.shape
        # 随机打乱波段顺序，确保噪声分配的随机性
        all_bands = np.random.permutation(range(B))
        pos = 0  # 波段分配位置指针
        
        # 为每种噪声类型分配波段并应用噪声
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            # 处理比例形式的波段数量（如0.3表示30%的波段）
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            
            # 获取当前噪声类型作用的波段
            bands = all_bands[pos:pos + num_band]
            pos += num_band  # 更新位置指针
            
            # 应用噪声
            out = noise_maker(out, bands)
            
        return clip(out, 0., 1.)


class _AddGaussNoniid(object):
    """
    内部使用的非iid高斯噪声生成器（支持指定波段）
    
    与AddGaussNoniid的区别在于可通过__call__方法的bands参数指定需要添加噪声的波段，
    主要用于与AddNoiseMixed配合使用。
    """

    def __init__(self, min_sigma, max_sigma, channel_first=True):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.channel_first = channel_first

    def __call__(self, img, bands):
        """
        向指定波段添加非iid高斯噪声
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)
            bands (list): 需要添加噪声的波段索引列表
            
        返回:
            np.ndarray: 添加噪声后的图像
        """
        # 为每个目标波段生成独立的标准差
        sigmas = np.random.uniform(self.min_sigma, self.max_sigma, len(bands)) / 255
        B, H, W = img.shape
        
        # 为指定波段添加噪声
        for i, sigma in zip(bands, sigmas):
            noise = np.random.randn(1, H, W) * sigma
            img[i:i+1, ...] = img[i:i+1, ...] + noise  # 添加噪声到当前波段
            
        return clip(img, 0., 1.)


class _AddNoiseImpulse(object):
    """
    脉冲噪声生成器（椒盐噪声）
    
    脉冲噪声表现为图像中随机出现的纯白（盐噪声）或纯黑（椒噪声）像素，
    通常由传感器故障或数据传输错误引起。
    """

    def __init__(self, amounts, s_vs_p=0.5):
        """
        初始化脉冲噪声生成器
        
        参数:
            amounts (list): 噪声比例范围（如[0.1, 0.3]表示10%-30%的像素被污染）
            s_vs_p (float): 盐噪声占脉冲噪声的比例（默认0.5，即盐和椒噪声各占一半）
        """
        self.amounts = np.array(amounts)  # 噪声比例数组
        self.s_vs_p = s_vs_p  # 盐噪声比例

    def __call__(self, input_img, bands):
        """
        向指定波段添加脉冲噪声
        
        参数:
            input_img (np.ndarray或torch.Tensor): 输入图像
            bands (list): 需要添加噪声的波段索引列表
            
        返回:
            np.ndarray: 添加脉冲噪声后的图像
        """
        # 转换为NumPy数组（若输入为PyTorch张量）
        if istensor(input_img):
            img = np.array(input_img.detach().cpu())
        elif isarray(input_img):
            img = input_img.copy()
        else:
            raise TypeError("输入图像必须是NumPy数组或PyTorch张量")
        
        # 为每个波段随机选择噪声比例
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        
        # 为指定波段添加脉冲噪声
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
            
        return clip(img, 0., 1.)

    def add_noise(self, image, amount, salt_vs_pepper):
        """
        向单波段图像添加脉冲噪声
        
        参数:
            image (np.ndarray): 单波段图像，形状为(H, W)
            amount (float): 噪声比例（0-1），表示被污染像素的比例
            salt_vs_pepper (float): 盐噪声占比（0-1）
        """
        out = image.copy()
        p = amount  # 噪声概率
        
        # 随机选择被污染的像素
        flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
        # 区分盐噪声和椒噪声
        salted = np.random.choice([True, False], size=image.shape, p=[salt_vs_pepper, 1 - salt_vs_pepper])
        peppered = ~salted  # 椒噪声像素为盐噪声像素的补集
        
        # 添加盐噪声（像素值设为1）和椒噪声（像素值设为0）
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """
    条纹噪声生成器
    
    条纹噪声表现为图像中出现的水平或垂直条纹，通常由传感器列/行响应不一致引起，
    是高光谱图像中常见的噪声类型。
    """

    def __init__(self, min_amount, max_amount):
        """
        初始化条纹噪声生成器
        
        参数:
            min_amount (float): 条纹数量占图像宽度的最小比例
            max_amount (float): 条纹数量占图像宽度的最大比例
            
        异常:
            AssertionError: 若max_amount <= min_amount则触发
        """
        assert max_amount > min_amount, "最大比例必须大于最小比例"
        self.min_amount = min_amount
        self.max_amount = max_amount        

    def __call__(self, img, bands):
        """
        向指定波段添加条纹噪声
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)
            bands (list): 需要添加噪声的波段索引列表
            
        返回:
            np.ndarray: 添加条纹噪声后的图像
        """
        B, H, W = img.shape
        
        # 为每个波段随机生成条纹数量
        num_stripe = np.random.randint(
            np.floor(self.min_amount * W),  # 最小条纹数
            np.floor(self.max_amount * W),  # 最大条纹数
            len(bands)
        )
        
        # 为指定波段添加条纹噪声
        for i, n in zip(bands, num_stripe):
            # 随机选择条纹位置
            loc = np.random.permutation(range(W))[:n]
            # 为每条条纹生成随机偏移值（[-0.25, 0.25]范围）
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25            
            # 应用条纹噪声
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
            
        return img


class _AddNoiseDeadline(object):
    """
    截止线噪声生成器
    
    截止线噪声表现为图像中某些列/行的像素值被置零，通常由传感器故障导致，
    是高光谱图像中一种典型的条带噪声。
    """

    def __init__(self, min_amount, max_amount):
        """
        初始化截止线噪声生成器
        
        参数:
            min_amount (float): 截止线数量占图像宽度的最小比例
            max_amount (float): 截止线数量占图像宽度的最大比例
            
        异常:
            AssertionError: 若max_amount <= min_amount则触发
        """
        assert max_amount > min_amount, "最大比例必须大于最小比例"
        self.min_amount = min_amount
        self.max_amount = max_amount        

    def __call__(self, img, bands):
        """
        向指定波段添加截止线噪声
        
        参数:
            img (np.ndarray): 输入图像，形状为(B, H, W)
            bands (list): 需要添加噪声的波段索引列表
            
        返回:
            np.ndarray: 添加截止线噪声后的图像
        """
        B, H, W = img.shape
        
        # 为每个波段随机生成截止线数量
        num_deadline = np.random.randint(
            np.ceil(self.min_amount * W),  # 最小截止线数
            np.ceil(self.max_amount * W),  # 最大截止线数
            len(bands)
        )
        
        # 为指定波段添加截止线噪声
        for i, n in zip(bands, num_deadline):
            # 随机选择截止线位置
            loc = np.random.permutation(range(W))[:n]
            # 将截止线位置的像素值置零
            img[i, :, loc] = 0
            
        return img


class AddNoiseComplex(AddNoiseMixed):
    """
    复杂混合噪声生成器（继承自AddNoiseMixed）
    
    可根据需要配置多种噪声组合，例如高斯噪声+脉冲噪声+条纹噪声等，
    用于模拟真实场景中复杂的噪声干扰。
    """
    def __init__(self, intensity, noniid=None):
        if noniid is None:
            noniid = [10,50]
        sig_min = noniid[0]
        sig_max = noniid[1]
        self.noise_bank = [
            _AddGaussNoniid(sig_min,sig_max),
            _AddNoiseStripe(intensity,intensity+0.1), 
            _AddNoiseDeadline(intensity,intensity+0.1),
            _AddNoiseImpulse([intensity])
        ]
        self.num_bands = [2/3, 1/3, 1/3, 1/3]
