import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from wo_STA import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from fold_2 import folds

def infer(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for eeg, feature, label in dataloader:
            eeg, feature, label = eeg.to(device), feature.to(device), label.long().to(device)

            # 前向传播
            outputs = model(feature, eeg)
            loss = criterion(outputs[2], label)

            # 累计损失
            total_loss += loss.item() * eeg.size(0)

            # 计算准确率
            _, predicted = outputs[2].max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy

######################
#  仅在原脚本最外部添加一个循环来测试所有模型
######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你训练时每2 epoch保存一次模型，可以把这些模型路径或 epoch列表列出来
# model_epochs = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
model_epochs = [1,2,3,4,5,6,7,8,9,10]
model_paths = [f"/root/autodl-tmp/project/STA_model/MI/2_model/2_model_epoch_{ep}.ph" for ep in model_epochs]

all_models_mean_acc = []  # 存储每个模型的10窗平均准确率
model_names = []           # 存储模型名(或epoch)

fold_id = 1
train_index = folds[fold_id]["train"]
test_index = folds[fold_id]["test"]
print("第 2 折的测试索引:", test_index)

for mp in model_paths:
    if not os.path.exists(mp):
        print(f"模型文件不存在: {mp}")
        continue

    print(f"\n正在测试模型: {mp}")
    # 加载模型
    model = MultiModalModel_woSTA()
    model.to(device)
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()

    # 和你原脚本类似，统计10个窗口的准确率
    window_acc_list = [[] for _ in range(10)]  # window_acc_list[b]存各被试acc

    # 每被试10窗口
    for a in test_index:
        for b in range(10):  # 10个时间窗
            eeg_file = f'/root/autodl-tmp/project/pt_testing_4D_data/MI/sub0{a+1}_test4d_0{b}.pt'
            fnirs_file = f'/root/autodl-tmp/project/BCI/fNIRS/pt_testing_4D_data/MI/sub0{a+1}_test4d_0{b}.pt'
            target_file= f'/root/autodl-tmp/project/test_target/MI/sub0{a+1}_test_label_0{b}.pt'

            # 检查文件是否存在
            if not os.path.exists(eeg_file) or not os.path.exists(fnirs_file) or not os.path.exists(target_file):
                print(f"数据文件不存在: {eeg_file}, {fnirs_file}, {target_file}")
                continue

            # 创建 Dataset
            dataset_i = MultimodalDataset(
                eeg_file_path=eeg_file,
                fnirs_file_path=fnirs_file,
                target_path=target_file,
                transform=False,
                target_transform=False
            )

            test_dataloader = DataLoader(
                dataset_i,
                shuffle=False,  # 测试不打乱
                num_workers=0,
                batch_size=Config.test_batch_size,
                drop_last=False
            )

            # 调用 infer 函数
            acc = infer(model, test_dataloader, device)
            window_acc_list[b].append(acc)

    # 计算每个窗口平均准确率
    window_accuracy = {}
    for b in range(10):
        if len(window_acc_list[b]) > 0:
            window_accuracy[b] = sum(window_acc_list[b]) / len(window_acc_list[b])
        else:
            window_accuracy[b] = 0.0

    # 画原先那张 "每个窗口的准确率" 图
    names = [str(i+1) for i in range(10)]  # ['1', '2', ..., '10']
    x = range(len(names))
    y_1 = [window_accuracy[b] for b in range(10)]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_1, color='red', marker='o', linestyle='-', label='Multimodal')
    plt.legend()  # 显示图例
    plt.xticks(x, names)
    plt.xlabel("Time Window (s)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Recognition Accuracy per Time Window\n{os.path.basename(mp)}")
    plt.ylim(0, 100)
    for i, acc in enumerate(y_1):
        plt.text(x[i], acc + 1, f"{acc:.2f}%", ha='center')
    plt.show()

    # 打印窗口准确率
    for b in range(10):
        print(f"Window {b+1}: Accuracy = {y_1[b]:.2f}%")

    # 计算平均准确率
    mean_acc = sum(y_1)/10.0
    print(f"模型 {os.path.basename(mp)} 的平均准确率= {mean_acc:.2f}%\n")

    # 记录
    all_models_mean_acc.append(mean_acc)
    model_names.append(os.path.basename(mp))

# 画 "各模型平均准确率" 图
plt.figure(figsize=(8,5))
x_models = range(len(all_models_mean_acc))
plt.plot(x_models, all_models_mean_acc, color='blue', marker='D', linestyle='-', label='Avg Acc (10 windows)')
plt.xticks(x_models, model_names, rotation=45)
plt.xlabel("Model File")
plt.ylabel("Average Accuracy (%)")
plt.ylim(0,100)
plt.title("Comparison: Each Model's Average Accuracy across 10 windows")
for i,acc in enumerate(all_models_mean_acc):
    plt.text(i, acc+1, f"{acc:.2f}%", ha='center')
plt.legend()
plt.tight_layout()
plt.show()