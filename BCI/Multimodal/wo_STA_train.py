import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from wo_STA import *
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from fold_2 import folds

set_seed(Config.seed)
all_datasets = []

fold_id = 1
train_index = folds[fold_id]["train"]
test_index = folds[fold_id]["test"]

print("第 2 折的训练索引:", train_index)

for i in train_index:
    eeg_path = f'/root/autodl-tmp/project/pt_training_4D_data/MA/sub0{i+1}_train4d.pt'
    fnirs_path = f'/root/autodl-tmp/project/BCI/fNIRS/pt_training_4D_data/MA/sub0{i+1}_train4d.pt'
    target_path = f'/root/autodl-tmp/project/train_target/MA/sub0{i+1}_train_target.pt'

    dataset_i = MultimodalDataset(
        eeg_file_path=eeg_path,
        fnirs_file_path=fnirs_path,
        target_path=target_path,
        transform=False,
        target_transform=False
    )
    all_datasets.append(dataset_i)

# 用 ConcatDataset 把所有被试数据拼接起来
dataset_all = ConcatDataset(all_datasets)
print("Total samples in combined dataset:", len(dataset_all))

# 2) 构建训练 DataLoader
train_dataloader = DataLoader(
    dataset_all,
    batch_size=Config.train_batch_size,  # 例如 8
    shuffle=True,
    num_workers=0,
    drop_last=True
)

# 3) 准备模型（单 GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel_woSTA().to(device)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4) 训练循环
losses = []
accuracies = []
model.train()

for epoch in range(Config.train_number_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    # 用 tqdm 包装 train_dataloader 来显示进度条
    loop = tqdm(train_dataloader,
                desc=f"Epoch [{epoch + 1}/{Config.train_number_epochs}]",
                leave=True)

    for batch_idx, (eeg, fnirs, label) in enumerate(loop):
        # 把数据放到 GPU 或 CPU
        eeg = eeg.to(device)
        fnirs = fnirs.to(device)
        label = label.long().to(device)

        # 前向传播
        outputs = model(fnirs, eeg)  # (y_fnirs, y_eeg, y_final)
        logits = outputs[2]  # 取 y_final 做二分类

        loss = criterion(logits, label)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item() * eeg.size(0)
        _, predicted = logits.max(dim=1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        # 更新进度条右侧信息：显示当前 batch 的 loss
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total
    losses.append(epoch_loss)
    accuracies.append(accuracy)

    print(f"[Epoch {epoch + 1}/{Config.train_number_epochs}] "
          f"Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}%")


    save_path = f"/root/autodl-tmp/project/STA_model/MA/2_model/2_model_epoch_{epoch + 1}.ph"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存: {save_path}")


# 训练结束后，可视化损失和准确率
plot_loss_accuracy(losses, accuracies, Config.train_number_epochs)

print("训练已完成")
