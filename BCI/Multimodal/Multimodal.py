import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random


# ## 用于配置的帮助类
class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 8  # 64
    test_batch_size = 8
    train_number_epochs = 30  # 100
    test_number_epochs = 20
    seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积算法的确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动寻找最适合当前配置的算法
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定hash种子


def plot_loss_accuracy(losses, accuracies, num_epochs):
    """
    绘制损失和准确率变化图（双坐标轴）。

    参数：
    - losses: 每个 epoch 的损失列表。
    - accuracies: 每个 epoch 的准确率列表。
    - num_epochs: 训练的总 epoch 数。
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 绘制损失曲线（左 y 轴）
    ax1.plot(range(1, num_epochs + 1), losses, marker='o', color='blue', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 绘制准确率曲线（右 y 轴）
    ax2 = ax1.twinx()
    ax2.plot(range(1, num_epochs + 1), accuracies, marker='o', color='orange', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 添加图例
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), bbox_transform=ax1.transAxes)

    # 设置标题
    plt.title('Loss and Accuracy over Epochs')

    # 展示图像
    plt.tight_layout()
    plt.show()


class MultimodalDataset(Dataset):
    def __init__(self, eeg_file_path,fnirs_file_path, target_path, transform=None, target_transform=None):
        self.eeg_file_path = eeg_file_path
        self.fnirs_file_path = fnirs_file_path
        self.target_path = target_path
        self.eeg_data = self.parse_data_file(eeg_file_path)
        self.fnirs_data = self.parse_data_file(fnirs_file_path)
        self.target = self.parse_target_file(target_path)

        #self.transform = transform
        #self.target_transform = target_transform

    def parse_data_file(self, file_path):

        data = torch.load(file_path)
        return np.array(data, dtype=np.float32)

    def parse_target_file(self, target_path):
        target = torch.load(target_path)  # 假设是一个可迭代对象
        target = np.array(target, dtype=np.float32)

        # 转换
        target[target == 16] = 0
        target[target == 32] = 1

        return target

    def __len__(self):

        return len(self.target)

    def __getitem__(self, index):
        eeg = self.eeg_data[index, :]
        fnirs = self.fnirs_data[index, :]
        target = self.target[index]

        #if self.transform:
            #item = self.transform(item)
        #if self.target_transform:
            #target = self.target_transform(target)

        return eeg,fnirs, target


# ------------------------ 模块定义 ------------------------ #

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        RFB模块 (Receptive Field Block)
        """
        super(RFB, self).__init__()

        # 分支1：1x1卷积 + 3x3卷积，无空洞率
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2：1x1卷积 + 1x3卷积 + 3x1卷积 + 3x3卷积（空洞率为 (3,3,1)）
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=(3, 3, 1), dilation=(3, 3, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支3：1x1卷积 + 1x5卷积 + 5x1卷积 + 3x3卷积（空洞率为 (5,5,1)）
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=(5, 5, 1), dilation=(5, 5, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 分支4：1x1卷积 + 1x7卷积 + 7x1卷积 + 3x3卷积（空洞率为 (7,7,1)）
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=(7, 7, 1), dilation=(7, 7, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 拼接后的通道压缩
        self.conv_cat = nn.Sequential(nn.Conv3d(64, out_channels, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0)),
                                      nn.BatchNorm3d(out_channels),
                                      nn.ReLU(inplace=True)
                                      )

        # 残差连接
        self.conv_residual = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(out_channels),
                                           nn.ReLU(inplace=True)
                                           )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多分支卷积操作
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        # 拼接所有分支输出
        x_cat = torch.cat([x1, x2, x3, x4], dim=2)

        # 通道压缩
        x_cat = self.conv_cat(x_cat)

        # 残差连接
        x_residual = self.conv_residual(x)
        x_out = x_cat + x_residual

        return self.relu(x_out)

class CBR(nn.Module):
    """
    CBR模块：Conv + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        """
        Gated Fusion模块
        """
        super(GatedFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.pool1 = nn.MaxPool3d(kernel_size=(13, 1, 1), stride=(13, 1, 1), padding=(0, 0, 0))

        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0)),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.pool2 =nn.MaxPool3d(kernel_size=(13, 1, 1),stride=(13, 1, 1),padding=(0, 0, 0))

        self.conv3 = nn.Sequential(nn.Conv3d(in_channels , in_channels, kernel_size=(2,1,1),stride=(2,1,1)),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(12, 1, 1), stride=(12, 1, 1)),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=2)  # depth维度拼接
        gate1 = self.sigmoid(self.pool1(self.conv1(cat)))
        gate2 = self.sigmoid(self.pool2(self.conv2(cat)))

        x2 = self.conv4(x2)
        gate1 = gate1 * x1
        gate2 = gate2 * x2

        gate1 = gate1 + x1
        gate2 = gate2 + x2

        out = torch.cat([gate1, gate2], dim=2)
        out = self.conv3(out)
        return out


class FAM(nn.Module):
    def __init__(self, in_channels):
        """
        FAM模块
        """
        super(FAM, self).__init__()
        # 最大池化层（在时间维度上进行池化）
        self.time_pool = nn.AdaptiveMaxPool3d((1, None, None))  # 将时间维度压缩为 1

        # 两层线性层
        self.fc1 = nn.Linear(in_channels, 32)  # 第一层全连接
        self.fc2 = nn.Linear(32, in_channels)  # 第二层全连接

        # 激活函数
        self.relu = nn.ReLU()

        self.conv = nn.Sequential(nn.Conv3d(in_channels , in_channels, kernel_size=1),
                                 nn.BatchNorm3d(in_channels),
                                 nn.ReLU(inplace=True)
                                 )

    def forward(self, x1, x2):
        # 1. 时间维度最大池化
        pooled = self.time_pool(x1)  # 形状为 (batch_size, channels, 1, height, width)
        pooled = pooled.squeeze(2)  # 去掉时间维度，变成 (batch_size, channels, height, width)

        # 2. 全连接层处理
        # 将 pooled 数据展平到 (batch_size * height * width, channels)
        pooled_flat = pooled.view(-1, x1.size(1))

        # 通过两层全连接
        fc_out = self.fc1(pooled_flat)  # 第一层全连接
        fc_out = self.relu(fc_out)
        fc_out = self.fc2(fc_out)  # 第二层全连接
        fc_out = self.relu(fc_out)

        # 将全连接结果 reshape 回 (batch_size, channels, height, width)
        fc_out = fc_out.view(x1.size(0), x1.size(1), 16, 16)

        # 3. 广播和逐元素相乘
        # 将 fc_out 广播回输入的形状 (batch_size, channels, time, height, width)
        fc_out_expanded = fc_out.unsqueeze(2).expand_as(x1)

        # 与输入逐元素相乘
        out = x1 * fc_out_expanded

        x_fam = torch.cat([out, x2], dim=2)  # 通道维度拼接
        return self.conv(x_fam)


class SpatialAttentionPooling(nn.Module):
    """
    SAP: Spatial Attention Pooling 模块
    """
    def __init__(self, in_channels, pool_size=(None, 1, 1)):
        """
        初始化SAP模块
        """
        super(SpatialAttentionPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(pool_size)  # 空间池化（平均池化）

    def forward(self, x):
        """
        前向传播
        """
        return self.pool(x)


class EfficientMultiScaleAttention(nn.Module):
    """
    Efficient Multi-Scale Attention 模块
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化Efficient Multi-Scale Attention模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(EfficientMultiScaleAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=1),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv5 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 [batch, channels, depth, height, width]
        :return: 多尺度特征融合后的张量
        """
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        return x1 + x3 + x5


class STA(nn.Module):
    """
    STA: Spatiotemporal Attention 模块
    """
    def __init__(self, in_channels, out_channels, pool_size=(None, 1, 1)):
        """
        初始化STA模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param pool_size: SAP池化窗口大小
        """
        super(STA, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sap = SpatialAttentionPooling(in_channels, pool_size)  # 空间注意力池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1),  # 全连接层
            nn.Sigmoid()  # 激活函数
        )
        self.efficient_attention = EfficientMultiScaleAttention(in_channels, out_channels)  # 多尺度注意力

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 [batch, channels, depth, height, width]
        :return: 输出张量
        """
        # SAP 过程
        sap_out = self.sap(x)  # [batch, channels, Depth, 1, 1]
        sap_out = sap_out.view(-1, sap_out.size(1))  # 展平为 [batch, channels]
        sap_out = self.fc(sap_out)  # 全连接输出 [batch, out_channels]
        sap_out = self.sigmoid(sap_out)
        sap_out = sap_out.reshape(x.size(0),1,-1,1,1)
        sap_out = sap_out.expand_as(x)
        # 多尺度注意力
        multi_scale_out = self.efficient_attention(x)  # [batch, out_channels, depth, height, width]

        # 融合
        attention_out = sap_out * multi_scale_out  # 元素级乘法
        out = attention_out + x  # 加法融合

        return out

# ------------------------ 主模型 ------------------------ #

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # RFB模块
        self.rfb_fnirs = RFB(in_channels=1, out_channels=64)
        self.rfb_eeg = RFB(in_channels=1, out_channels=64)

        # CBR模块
        self.cbr_fnirs_1 = CBR(64, 64)
        self.cbr_fused_1 = CBR(128, 128)
        self.cbr_eeg_1 = CBR(64, 128)

        self.cbr_fnirs_2 = CBR(64, 128)
        self.cbr_fused_2 = CBR(128, 128)
        self.cbr_eeg_2 = CBR(128, 128)

        self.cbr_fnirs_3 = CBR(128, 256)
        self.cbr_eeg_3 = CBR(128, 128)

        self.cbr_fnirs_4 = CBR(256,128)

        # Gated Fusion模块
        self.gated_fusion = GatedFusion(in_channels=64)

        # FAM模块
        self.fam_fused_1 = FAM(in_channels=128)
        self.fam_fused_2 = FAM(in_channels=128)
        self.fam_eeg_1 = FAM(in_channels=128)
        self.fam_eeg_2 = FAM(in_channels=128)

        # STA模块
        self.sta_fnirs = STA(in_channels=128,out_channels=128)
        self.sta_eeg = STA(in_channels=128,out_channels=128)
        self.sta_fused = STA(in_channels=128,out_channels=128)

        # 最后的全连接层
        # 2分类
        self.fc_pool_fnirs = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
                                           nn.BatchNorm3d(64),
                                           nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool3d((60, 1, 1)),
                                           )
        self.fc_fnirs = nn.Sequential(nn.Dropout(p=0.5),
                                      nn.Linear(3840, 480),
                                      nn.Linear(480, 60),
                                      nn.BatchNorm1d(60),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(60, 2),
                                      )

        self.fc_pool_eeg = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
                                         nn.BatchNorm3d(64),
                                         nn.ReLU(inplace=True),
                                         nn.AdaptiveAvgPool3d((480, 1, 1)),
                                        )
        self.fc_eeg =  nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Linear(30720, 3840),
                                     nn.Linear(3840, 480),
                                     nn.BatchNorm1d(480),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(480, 60),
                                     nn.Linear(60, 2),
                                      )

        self.fc_pool_fused = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
                                           nn.BatchNorm3d(64),
                                           nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool3d((180, 1, 1)),
                                           )
        self.fc_fused =  nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(11520, 1440),
                                       nn.Linear(1440, 180),
                                       nn.BatchNorm1d(180),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(180, 45),
                                       nn.Linear(45, 2),
                                       )

        # 可学习的alpha
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_fnirs, x_eeg):
        N = x_fnirs.size(0)
        x_eeg = x_eeg.reshape([N,1,360,16,16])
        # RFB模块
        x_fnirs = self.rfb_fnirs(x_fnirs)
        x_eeg = self.rfb_eeg(x_eeg)

        # Gated Fusion
        fused = self.gated_fusion(x_fnirs, x_eeg)


        # CBR模块
        x_fnirs = self.cbr_fnirs_1(x_fnirs)
        x_eeg = self.cbr_eeg_1(x_eeg)

        fused = torch.cat([fused, x_fnirs], dim=1)
        x_fnirs = self.cbr_fnirs_2(x_fnirs)

        # FAM模块
        fused = self.fam_fused_1(x_fnirs,fused)
        x_eeg = self.fam_eeg_1(x_fnirs, x_eeg)

        fused = self.cbr_fused_1(fused)
        x_eeg = self.cbr_eeg_2(x_eeg)
        x_fnirs = self.cbr_fnirs_3(x_fnirs)

        x_fnirs = self.cbr_fnirs_4(x_fnirs)
        fused = self.fam_fused_2(x_fnirs,fused)
        x_eeg = self.fam_eeg_2(x_fnirs, x_eeg)

        fused = self.cbr_fused_2(fused)
        torch.cuda.empty_cache()
        x_eeg = self.cbr_eeg_3(x_eeg)
        x_fnirs = self.sta_fnirs(x_fnirs)

        x_eeg = self.sta_eeg(x_eeg)
        fused = self.sta_fused(fused)

        x_fnirs = self.fc_pool_fnirs(x_fnirs)
        x_fnirs = x_fnirs.view(x_fnirs.size(0), -1)
        x_fnirs = self.fc_fnirs(x_fnirs)
        #y_fnirs = F.softmax(x_fnirs,dim=1)
        y_fnirs = x_fnirs

        fused = self.fc_pool_fused(fused)
        fused = fused.view(fused.size(0), -1)
        fused = self.fc_fused(fused)

        x_eeg = self.fc_pool_eeg(x_eeg)
        x_eeg = x_eeg.view(x_eeg.size(0), -1)
        x_eeg = self.fc_eeg(x_eeg)

        #y_eeg = F.softmax(x_eeg,dim=1)
        y_eeg = x_eeg
        #y_fused = F.softmax(fused,dim=1)
        y_fused = fused


        # 融合输出
        y_final = self.alpha * y_fused + (1 - self.alpha) * y_eeg

        return y_fnirs, y_eeg, y_final

