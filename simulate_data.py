# -*- coding: utf-8 -*-
"""
高光谱图像(HSI)模拟数据生成脚本

该脚本用于从原始高光谱图像生成带有不同噪声的模拟数据，主要包括以下步骤：
1. 加载原始高光谱图像(HSI)并进行预处理
2. 模拟全色图像(PAN)的生成
3. 定义多种噪声模型
4. 将图像切割为固定大小的 patches
5. 为每个 patch 添加不同类型的噪声并保存结果

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

# 导入必要的库
import os  # 用于文件和目录操作
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 解决PyTorch多线程冲突问题
import utils.noise_model as N  # 导入自定义的噪声模型工具
from scipy.io import loadmat, savemat  # 用于读取和保存.mat格式文件
import numpy as np  # 用于数值计算和数组操作
from utils import *  # 导入自定义工具函数


# 加载高光谱图像(HSI)
# 从.mat文件加载HSI数据，转换为float32类型并归一化到[0,1]范围
HSI = loadmat('data/Dongying_1_1.mat')['HSI'].astype(np.float32) / 1023.  # 形状为[高度, 宽度, 波段数]
height, width, nband = HSI.shape  # 获取HSI的高度、宽度和波段数

# 定义参数
ratio = 4  # 下采样比例
patch_size = 256  # 图像块的大小
# 计算宽度方向上可切割的patch数量
num_patches_width = (width - patch_size) // patch_size + 1
# 计算高度方向上可切割的patch数量
num_patches_height = (height - patch_size) // patch_size + 1

# 对HSI进行中心裁剪，确保能被均匀分割为多个patch
HSI = center_crop(HSI, patch_size * num_patches_width)
# 打印加载的HSI信息
print(f'加载的HSI形状为 {HSI.shape}, 数据类型为 {HSI.dtype}, 取值范围为 [%.2f,%.2f]' 
      % (HSI.min(), HSI.max()))

# 模拟全色图像(PAN)
# 加载IKONOS卫星的光谱响应函数(SRF)
srf = loadmat('utils/srf/ikonos_spec_resp.mat')['ikonos_sp'][:, 1]
# 定义与HSI重叠的波段索引
overlap_band = [23, 26, 29, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61,
                64, 67, 70, 74, 76, 79, 82, 85, 88, 91, 94, 97, 100,
                103, 106, 109, 112, 115, 118]
# 提取与HSI重叠的SRF部分
srf = srf[overlap_band]
# 归一化SRF，使其总和为1
srf = srf / sum(srf)
# 调整SRF的形状以匹配HSI的维度，便于广播运算
srf = np.reshape(srf, [1, 1, nband])
# 通过HSI与SRF的加权求和生成PAN图像
PAN = np.sum(srf * HSI, -1)
# 打印生成的PAN图像信息
print(f'生成的PAN图像形状为 {PAN.shape}, 数据类型为 {PAN.dtype}, 取值范围为 [%.2f,%.2f]' 
      % (PAN.min(), PAN.max()))

# 定义噪声模型字典
# 包含不同类型和强度的噪声：高斯噪声、非平稳高斯噪声、混合噪声等
noise_models = {
    'g10': N.AddGauss(10),          # 高斯噪声，标准差10
    'g30': N.AddGauss(30),          # 高斯噪声，标准差30
    'gni': N.AddGaussNoniid(10, 50),# 非独立同分布高斯噪声，标准差范围10-50
    'mix15': N.AddNoiseComplex(0.15),# 复杂混合噪声，强度0.15
    'mix35': N.AddNoiseComplex(0.35),# 复杂混合噪声，强度0.35
    'mix55': N.AddNoiseComplex(0.55) # 复杂混合噪声，强度0.55
}

# 为每种噪声模型创建对应的存储目录
for nm in list(noise_models.keys()):
    os.makedirs(f'data/{nm}', exist_ok=True)

# 切割图像为patches并添加噪声
# 遍历所有高度方向的patch
for i in range(num_patches_height):
    # 遍历所有宽度方向的patch
    for j in range(num_patches_width):
        # 生成基于当前patch位置的随机种子，确保结果可复现
        seed = int(f'{i}{j}')
        
        # 计算当前patch的起始和结束坐标
        start_x = j * patch_size
        end_x = start_x + patch_size
        start_y = i * patch_size
        end_y = start_y + patch_size
        
        # 提取当前patch
        I_GT = HSI[start_y:end_y, start_x:end_x, :]  # 原始高光谱图像patch（无噪声）
        I_PAN = PAN[start_y:end_y, start_x:end_x]    # 全色图像patch
        # 生成低分辨率高光谱图像（先转换为tensor，下采样，再转换回array）
        I_LRHS = tensor2array(downsample_hsi(array2tensor(I_GT, 'cuda'), ratio))
        
        # 为当前patch添加不同类型的噪声并保存
        for nm in list(noise_models.keys()):
            # 设置随机种子，确保同一种子下的噪声模式一致
            setup_seed(seed)
            # 对低分辨率高光谱图像添加噪声：需要先转置维度以匹配噪声模型的输入要求
            N_LRHS = noise_models[nm](I_LRHS.copy().transpose(2, 0, 1)).transpose(1, 2, 0)
            
            # 检查噪声添加后的数据范围是否超出[0,1]
            if N_LRHS.min() < 0 or N_LRHS.max() > 1:
                print(f'噪声 {nm} 导致数据范围超出：{N_LRHS.min()}, {N_LRHS.max()}')
            
            # 保存处理后的patch数据（转换回0-1023范围的uint16类型）
            savemat(
                f'data/{nm}/Dongying_{i}_{j}.mat',
                {
                    'I_GT': (I_GT * 1023).astype(np.uint16),    # 原始高光谱图像
                    'I_PAN': (I_PAN * 1023).astype(np.uint16),  # 全色图像
                    'I_LRHS': (I_LRHS * 1023).astype(np.uint16),# 低分辨率高光谱图像
                    'N_LRHS': (N_LRHS * 1023).astype(np.uint16) # 带噪声的低分辨率高光谱图像
                }
            )