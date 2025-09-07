# -*- coding: utf-8 -*-
"""
高光谱图像重建主程序（UHipandas方法）
功能：加载带噪声的高光谱数据，通过深度学习模型（GDN、GSRN、PRN）进行去噪和超分辨率重建，
      保存结果并输出评估指标和运行时间

文章：
Shuang Xu, Zixiang Zhao, Haowen Bai, Chang Yu, Jiangjun Peng, Xiangyong Cao, Deyu Meng, 
Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image. 
ICCV, 2025.
"""

import os
# 解决PyTorch多线程冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import utils  # 自定义工具函数
import torch
import torch.nn as nn
import kornia as K  # 计算机视觉库（提供Sobel算子等）
from scipy.io import loadmat, savemat  # 读写.mat文件
from time import time  # 计时
from model import GDN, GSRN, PRN  # 导入自定义模型
from matplotlib import pyplot as plt  # 图像显示

# 定义噪声场景列表
noise_case = ['g10', 'g30', 'gni', 'mix15', 'mix35', 'mix55']

# 遍历每个噪声场景
for nc in noise_case:
    print('noise case: ', nc)
    # 创建结果保存路径
    savepath = 'result/UHipandas/%s'%(nc)
    os.makedirs(savepath, exist_ok=True)
    
    # 根据噪声场景设置正则化参数sigma
    if nc.startswith('g'):  # 高斯噪声（g10, g30）
        sig = nc[1:]
        try:
            sig = int(sig)
        except:
            sig = 30  # 默认值
    elif nc.startswith('mix'):  # 混合噪声（mix15, mix35）
        sig = nc[3:]
        sig = 0.5 * int(sig) + 15  # 计算混合噪声的sigma
    
    # 遍历9x9=81幅测试图像
    for i in range(9):
        for j in range(9):
            filename = 'Dongying_%d_%d.mat'%(i,j)  # 图像文件名
            data_path = 'data/%s/%s'%(nc,filename)  # 数据路径
            print('data: ', filename)

            # 加载数据
            data = loadmat(data_path)
            max_val = 2**10 - 1  # 10位量化最大值（1023）
            # 读取并归一化数据到[0,1]
            I_GT_np = data['I_GT'].astype('float32') / max_val  # 真实高光谱图像
            I_LRHS_np = data['I_LRHS'].astype('float32') / max_val  # 低分辨率带噪高光谱
            I_PAN = data['I_PAN'].astype('float32') / max_val  # 全色图像（高分辨率）
            N_LRHS = data['N_LRHS'].astype('float32') / max_val  # 带噪声的低分辨率高光谱
            ratio = 4  # 超分倍数（4x）
            
            # 数据转换为PyTorch张量（并转移到GPU）
            Input = utils.array2tensor(N_LRHS, 'cuda')  # 输入噪声数据
            HRPAN = utils.array2tensor(I_PAN[..., None], 'cuda')  # 高分辨率全色图（添加通道维度）
            LRPAN = utils.imresize(HRPAN, 1/ratio)  # 下采样得到低分辨率全色图
            
            # 超参数设置
            lr = 1e-3  # 学习率
            num_epoch = [400, 600]  # 预训练和训练的 epoch 数
            seed = 8888  # 随机种子（保证可复现性）
            smooth_coef = 0.99  # 移动平均系数（用于结果平滑）
            reg_sigma = 0.0055 * sig  # 正则化噪声的标准差
            nbands = Input.shape[1]  # 高光谱波段数
            
            # 初始化设置
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备（GPU优先）
            torch.backends.cudnn.enabled = True  # 启用CuDNN加速
            torch.backends.cudnn.benchmark = True  # 自动优化卷积算法
            utils.setup_seed(seed)  # 设置随机种子
            
            # 定义模型、优化器和损失函数
            # 去噪模型（GDN）
            DeModel = GDN(
                rank=int(3/32 * nbands),  # 秩参数
                hsi_c=nbands,  # 高光谱通道数
                pan_c=1,  # 全色图通道数
                n_feat=128,  # 特征通道数
                layerU=2,  # 网络层数
                act=nn.ReLU(True),  # 激活函数
                bias=True,  # 使用偏置
                sigmoid_V=True,  # 对V使用sigmoid
                softmax_U=True  # 对U使用softmax
            ).to(device)
            
            # 超分模型（GSRN）
            SRModel = GSRN(
                rank=int(12/32 * nbands),
                hsi_c=nbands,
                pan_c=1,
                n_feat=128,
                layerU=2,
                act=nn.ReLU(True),
                bias=True,
                sigmoid_V=True,
                softmax_U=True
            ).to(device)
            
            # 全色模拟模型（PRN）
            PanModel = PRN(
                hsi_c=nbands,
                pan_c=1,
                n_feat=128,
                layer=5,
                act=nn.ReLU(True)
            ).to(device)
            
            # 优化器（Adam）
            optimizer = torch.optim.Adam(
                [{'params': DeModel.parameters()},
                 {'params': SRModel.parameters()},
                 {'params': PanModel.parameters()}],
                lr=lr
            )
            
            # 损失函数（L1和MSE）
            loss_fn1 = nn.L1Loss()
            loss_fn2 = nn.MSELoss()
            
            # 模型训练模式
            DeModel.train()
            SRModel.train()
            PanModel.train()
            
            # 预训练阶段
            t0 = time()  # 计时开始
            for epoch in range(int(num_epoch[0])):
                optimizer.zero_grad()  # 清空梯度
                
                # 复制输入数据（避免修改原始数据）
                working_Input = Input.clone()
                working_LRPAN = LRPAN.clone()
                
                # 去噪分支：添加正则化噪声，模型输出去噪结果
                net_input = working_Input + torch.randn_like(working_Input) * reg_sigma
                lrhs_hat, U, V = DeModel(net_input, working_LRPAN) 
                loss = loss_fn1(lrhs_hat, working_Input)  # 去噪损失
                
                # 超分一致性分支：下采样后再超分，约束一致性
                dshs = utils.imresize(utils.downsample_hsi(lrhs_hat, ratio), ratio)
                lrhs_hat_hat, U, V = SRModel(dshs, working_LRPAN)
                loss += loss_fn2(lrhs_hat, lrhs_hat_hat)  # 一致性损失
                
                # 全色模拟分支：约束低分辨率全色图的梯度一致性
                lrpan_hat = PanModel(lrhs_hat) 
                loss += loss_fn1(
                    K.filters.sobel(lrpan_hat),  # 预测全色图的梯度
                    K.filters.sobel(working_LRPAN)  # 真实全色图的梯度
                )  # 梯度损失
                
                # 反向传播和参数更新
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()  # 清空GPU缓存

            # 预训练结束，保存中间结果
            lrhs_hat0 = lrhs_hat.detach()
            pretraining_time = time() - t0  # 预训练时间
            
            # 输出预训练阶段的去噪效果
            print('Pretraining Stage-> Denoising performance on LRHS: ', 
                  ['%.4f'%(m) for m in utils.HSI_metrics(
                      I_LRHS_np, 
                      utils.tensor2array(lrhs_hat0, True)
                  )] )
            
            # 训练阶段
            metric = []
            output = utils.tensor2array(utils.imresize(lrhs_hat0, ratio))  # 初始化输出（预训练结果上采样）
            t0 = time()  # 计时开始
            # 动态调整训练epoch数（噪声越大，epoch越少）
            for epoch in range(int(num_epoch[1] * (1.2 - sig * 0.02))):
                optimizer.zero_grad()
                
                # 复制输入数据
                working_Input = Input.clone()
                working_LRPAN = LRPAN.clone()
                working_HRPAN = HRPAN.clone()
                
                # 去噪分支
                net_input = (working_Input + torch.randn_like(working_Input) * reg_sigma).clip(0, 1)
                lrhs_hat, U, V = DeModel(net_input, working_LRPAN) 
                loss = loss_fn1(lrhs_hat, working_Input)  # 去噪损失
                
                # 超分分支：上采样后再超分，约束下采样一致性
                ushs = utils.imresize(lrhs_hat, ratio)
                hrhs_hat, U, V = SRModel(ushs, working_HRPAN)
                loss += loss_fn2(
                    lrhs_hat, 
                    utils.downsample_hsi(hrhs_hat, ratio)
                )  # 下采样一致性损失

                # 全色模拟分支：约束高低分辨率全色图的梯度一致性
                hrpan_hat = PanModel(hrhs_hat)  # 高分辨率全色预测
                lrpan_hat = PanModel(lrhs_hat)  # 低分辨率全色预测
                # 高分辨率梯度损失
                loss += loss_fn1(
                    K.filters.sobel(hrpan_hat), 
                    K.filters.sobel(working_HRPAN)
                )
                # 低分辨率梯度损失
                loss += loss_fn1(
                    K.filters.sobel(lrpan_hat), 
                    K.filters.sobel(working_LRPAN)
                )
                
                # 反向传播和更新
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                # 转换为numpy数组（并裁剪到[0,1]）
                hrhs_hat_np = utils.tensor2array(hrhs_hat, clip=True)
                
                # 移动平均平滑结果（减少波动）
                if output is None:
                    output = hrhs_hat_np.copy()
                else:
                    output = smooth_coef * output + (1 - smooth_coef) * hrhs_hat_np
            
            # 计算最终重建结果的评估指标
            metric.append(utils.HSI_metrics(I_GT_np, output))
            print('Training stage-> Hypandas performance: ', 
                  '%04d'%(epoch + 1), 
                  ['%.4f'%(m) for m in metric[-1]] )
            training_time = time() - t0  # 训练时间

            # 输出总耗时
            print('Elapsed time = %f (Pretrain) +  %f (Train) = %f s.'
                  %(pretraining_time, training_time, pretraining_time + training_time))
            
            # 保存重建结果（转换回10位量化）
            savemat(
                os.path.join(savepath, filename), 
                {'output': (1023 * output).astype('uint16')}
            )
            # 显示重建结果
            img_recon = output.copy()
            utils.rsshow(output)
            plt.show()