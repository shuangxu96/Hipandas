# -*- coding: utf-8 -*-
"""
遥感图像质量评估指标计算模块

该模块实现了高光谱图像（HSI）融合、去噪和超分辨率任务中常用的定量评估指标，
包括峰值信噪比（PSNR）、结构相似性（SSIM）、光谱角映射（SAM）和相对全局误差（ERGAS），
支持NumPy数组和PyTorch张量两种数据格式的输入，可直接用于模型训练和结果评估。

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import numpy as np
# 从skimage库导入峰值信噪比和结构相似性的基础实现
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import torch
import torch.nn as nn


def PSNR3D(x, y):
    """
    计算3D高光谱图像的峰值信噪比（PSNR）
    
    PSNR是衡量图像失真程度的经典指标，值越高表示图像质量越好。
    对于高光谱图像，通过计算每个光谱波段的PSNR后取平均值得到最终结果。
    
    计算公式：
    PSNR = 10 * log10(MAX^2 / MSE)
    其中，MAX为图像最大像素值（此处为255），MSE为均方误差
    
    参数:
        x (np.ndarray): 预测图像，形状为(H, W, C)，H为高度，W为宽度，C为光谱波段数
        y (np.ndarray): 参考图像（真实值），形状需与x保持一致
    
    返回:
        float: 平均PSNR值（dB）
    """
    psnr_val = 0.  # 初始化PSNR总和
    # 将输入图像从[0,1]范围转换为[0,255]的uint8格式（符合PSNR计算惯例）
    x = (255. * x).astype(np.uint8)
    y = (255. * y).astype(np.uint8)
    
    # 遍历每个光谱波段计算PSNR并累加
    for i in range(x.shape[-1]):
        psnr_val += psnr_fn(x[:, :, i], y[:, :, i])
    
    # 返回平均PSNR值
    return psnr_val / x.shape[-1]  


def SSIM3D(x, y):
    """
    计算3D高光谱图像的结构相似性（SSIM）
    
    SSIM通过比较图像的亮度、对比度和结构信息评估相似度，取值范围为[-1,1]，
    越接近1表示两图像结构越相似。对于高光谱图像，计算每个波段的SSIM后取平均值。
    
    参数:
        x (np.ndarray): 预测图像，形状为(H, W, C)
        y (np.ndarray): 参考图像（真实值），形状需与x保持一致
    
    返回:
        float: 平均SSIM值
    """
    ssim_val = 0.  # 初始化SSIM总和
    # 将输入图像从[0,1]范围转换为[0,255]的uint8格式
    x = (255. * x).astype(np.uint8)
    y = (255. * y).astype(np.uint8)
    
    # 遍历每个光谱波段计算SSIM并累加
    for i in range(x.shape[-1]):
        ssim_val += ssim_fn(x[:, :, i], y[:, :, i])
    
    # 返回平均SSIM值
    return ssim_val / x.shape[-1]  


def SAM(x, y):
    """
    计算光谱角映射（SAM）指标
    
    SAM用于衡量高光谱图像中对应像素光谱向量之间的角度，角度越小表示光谱一致性越好。
    该指标能有效评估融合/超分辨率结果的光谱保真度。
    
    计算公式：
    SAM = arccos( (x·y) / (||x||·||y||) ) * (180/π)
    其中·表示点积，||·||表示L2范数
    
    参数:
        x (np.ndarray): 预测图像的光谱向量，形状为(H, W, C)
        y (np.ndarray): 参考图像的光谱向量，形状需与x保持一致
    
    返回:
        float: 平均光谱角（度）
    """
    HH, WW, CC = x.shape  # 获取图像高度、宽度和波段数
    # 将光谱图像展平为像素-光谱向量形式 (H*W, C)
    x = x.reshape(HH * WW, CC)
    y = y.reshape(HH * WW, CC)
    
    # 计算光谱向量的点积
    dot_product = np.sum(x * y, axis=-1)
    # 计算光谱向量的L2范数乘积（加小常数避免除零）
    norm_product = np.sqrt(
        (np.sum(x * x, axis=-1) + 1e-6) * 
        (np.sum(y * y, axis=-1) + 1e-6)
    )
    
    # 计算余弦相似度并截断到[-1,1]范围（避免数值误差导致的计算问题）
    cos_theta = np.clip(dot_product / norm_product, 0., 1.)
    # 计算弧度角并转换为角度
    sam_angle = np.arccos(cos_theta)
    sam_angle = np.mean(sam_angle) * (180 / np.pi)
    
    return sam_angle


class SAMLoss(nn.Module):
    """
    光谱角映射（SAM）损失函数的PyTorch实现
    
    用于在模型训练过程中监督光谱一致性，继承自PyTorch的nn.Module，
    支持批量处理和反向传播计算。
    """
    def __init__(self, reduction='mean'):
        """
        初始化SAMLoss
        
        参数:
            reduction (str): 损失聚合方式，可选'mean'（默认，取平均值）、
                            'sum'（求和）或None（返回原始损失）
        """
        super().__init__()
        self.reduction = reduction
        self.pi = np.pi  # 圆周率常数

    def forward(self, outputs, labels):
        """
        前向传播计算损失
        
        参数:
            outputs (torch.Tensor): 模型输出的预测图像，形状为(B, C, H, W)
            labels (torch.Tensor): 真实标签图像，形状需与outputs一致
        
        返回:
            tuple: (聚合后的损失值, 每个样本的原始损失)
        """
        # 计算输出和标签的L2范数（加小常数避免除零）
        norm_outputs = torch.sum(outputs * outputs, dim=1) + 1e-6
        norm_labels = torch.sum(labels * labels, dim=1) + 1e-6
        # 计算光谱向量点积
        scalar_product = torch.sum(outputs * labels, dim=1)
        # 计算范数乘积
        norm_product = torch.sqrt(norm_outputs * norm_labels)
        
        # 展平空间维度以便批量计算
        scalar_product = torch.flatten(scalar_product, 1, 2)
        norm_product = torch.flatten(norm_product, 1, 2)
        
        # 计算余弦相似度并截断到[0,1]范围
        ratio = torch.clamp(scalar_product / norm_product, 0, 1)
        # 转换为角度相关的损失（使用(1-ratio)^2近似角度损失）
        angle = (1 - torch.mean(ratio, dim=1)).pow(2)
        
        # 根据指定方式聚合损失
        if self.reduction == 'mean':
            sam_loss = torch.mean(angle)
        elif self.reduction == 'sum':
            sam_loss = torch.sum(angle)
        elif self.reduction is None:
            sam_loss = None
        
        return sam_loss, angle


def ERGAS(img_fake, img_real, scale=1):
    """
    计算相对全局误差（ERGAS）
    
    ERGAS是评估高光谱图像融合质量的专用指标，结合了空间分辨率差异的影响，
    值越小表示融合效果越好。
    
    计算公式：
    ERGAS = (100 / scale) * sqrt( (1/C) * sum( MSE_i / (mean_i)^2 ) )
    其中scale为分辨率比例，C为波段数，MSE_i为第i波段的均方误差，mean_i为第i波段的均值
    
    参数:
        img_fake (np.ndarray): 预测图像，2D形状为(H, W)或3D形状为(H, W, C)
        img_real (np.ndarray): 参考图像，形状需与img_fake一致
        scale (int, optional): 空间分辨率比例（高分辨率/低分辨率），默认1
    
    返回:
        float: ERGAS值
    
    异常:
        ValueError: 当输入图像形状不一致或维度错误时抛出
    """
    # 检查输入图像形状是否一致
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    # 转换为双精度浮点数以提高计算精度
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    
    # 处理2D单波段图像
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()  # 计算参考图像均值
        mse = np.mean((img_fake_ - img_real_) ** 2)  # 计算均方误差
        # 计算ERGAS（添加极小值避免除零）
        return 100 / scale * np.sqrt(mse / (mean_real **2 + np.finfo(np.float64).eps))
    
    # 处理3D多波段图像
    elif img_fake_.ndim == 3:
        # 计算每个波段的均值和均方误差
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        # 计算ERGAS（添加极小值避免除零）
        return 100 / scale * np.sqrt(
            (mses / (means_real **2 + np.finfo(np.float64).eps)).mean()
        )
    
    # 处理无效维度
    else:
        raise ValueError('Wrong input image dimensions.')


def print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'ERGAS']):
    """
    格式化打印评估指标结果
    
    以表格形式展示多个样本的评估指标，便于结果对比和分析。
    
    参数:
        metrics (np.ndarray): 评估指标数组，形状为(N, M)，N为样本数，M为指标数
        col_labels (list, optional): 指标名称列表，长度需与M一致，默认['PSNR', 'SSIM', 'ERGAS']
    """
    num_row, num_col = metrics.shape  # 获取样本数和指标数
    # 打印表头
    first_row = 'Metrics \t '
    for i in range(num_col):
        first_row += '%s \t    ' % (col_labels[i])
    print(first_row)
    
    # 打印每行数据
    for j in range(num_row):
        temp_row = '%7d \t' % (j + 1)  # 样本索引
        for i in range(num_col):
            temp_row += '%2.3f \t' % (metrics[j, i])  # 指标值（保留3位小数）
        print(temp_row)


def Convert3D(x):
    """
    将高光谱图像转换为标准3D格式 (H, W, C)
    
    处理可能的高维输入，确保后续指标计算的维度一致性。
    
    参数:
        x (np.ndarray): 输入图像，可包含批次维度或其他冗余维度
    
    返回:
        np.ndarray: 转换后的3D图像，形状为(H, W, C)
    """
    data_shape = x.shape
    # 将除前两维外的维度合并为光谱维度
    new_shape = [data_shape[0], data_shape[1], np.prod(data_shape[2:])]
    return np.reshape(x, new_shape)


def HSI_metrics(gt_img, test_img):
    """
    高光谱图像综合评估指标计算函数
    
    一站式计算PSNR、SSIM、ERGAS和SAM四个关键指标，返回标准化结果。
    
    参数:
        gt_img (np.ndarray): 参考图像（真实值），像素值范围[0,1]
        test_img (np.ndarray): 测试图像（预测值），像素值范围[0,1]，形状需与gt_img一致
    
    返回:
        list: 包含四个指标的列表，顺序为[PSNR, SSIM, ERGAS, SAM]
    """
    # 统一转换为3D格式 (H, W, C)
    gt_img = Convert3D(gt_img)
    test_img = Convert3D(test_img)
    
    # 依次计算各指标
    psnr_val = PSNR3D(gt_img, test_img)
    ssim_val = SSIM3D(gt_img, test_img)
    ergas_val = ERGAS(gt_img, test_img)
    sam_val = SAM(gt_img, test_img)
    
    return [psnr_val, ssim_val, ergas_val, sam_val]