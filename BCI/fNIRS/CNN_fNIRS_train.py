import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from CNN_fNIRS import *
import pandas as pd
import os


def infer(model,dataset,device):
    model.eval()
    correct = 0
    tot = 0
    accuracy = 0
    with torch.no_grad():
        for data in dataset:
            item, target = data
            item, target = item.to(device), target.to(device)

            output = net(item)  # 输出
            predicted = torch.argmax(output, 1)
            correct += (predicted == target).sum().item()
            tot += target.size(0)  # total += target.size
            accuracy = np.array(correct/ tot)
    return accuracy


for i in range(7,30):
    fNIRSnetdata = CNNNetDataset(file_path='/Users/lihao/PythonCode/BCI/fNIRS/pt_training_4D_data/sub0'+str(i)+'_train4d.pt',
                            target_path='/Users/lihao/PythonCode/BCI/fNIRS/train_target/sub0'+str(i)+'train_target.pt',
                            transform=False, target_transform=False)

    train_dataloader = DataLoader(fNIRSnetdata, shuffle=True, num_workers=0, batch_size=Config.train_batch_size,
                                drop_last=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = CNNNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.MultiMarginLoss()
    # optimizer = optim.SGD(net.parameters(),lr=0.8)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    counter = []
    loss_history = []
    iteration_number = 0
    train_correct = 0
    total = 0
    train_accuracy = []
    correct = 0
    total = 0
    classnum = 4
    accuracy_history = []

    net.train()

    for epoch in range(0, Config.train_number_epochs):
        for x, data in enumerate(train_dataloader, 0):  # enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
            item, target = data
            item, target = item.to(device), target.to(device)

            optimizer.zero_grad()  # grad归零
            output = net(item)  # 输出
            loss = criterion(output, target.long())  # 算loss,target原先为Tensor类型，指定target为long类型即可。
            loss.backward()  # 反向传播算当前grad
            optimizer.step()  # optimizer更新参数
            # 求ACC标准流程
            predicted = torch.argmax(output, 1)
            train_correct += (predicted == target).sum().item()
            total += target.size(0)  # total += target.size
            train_accuracy = train_correct / total
            train_accuracy = np.array(train_accuracy)

            if x % 10 == 0:  # 每10个epoch输出一次结果
                print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format
                    (epoch, train_accuracy.item(), loss.item()))
            iteration_number += 1
            counter.append(iteration_number)
            accuracy_history.append(train_accuracy.item())
            loss_history.append(loss.item())
            # torch.save(net.state_dict(), "/Users/lihao/PythonCode/model/The_"+str(i)+"_"+str(x)+"train.EEGNet.ph")

    show_plot(counter, accuracy_history, loss_history)

# 保存模型
torch.save(net.state_dict(), "/Users/lihao/PythonCode/BCI/fNIRS/model/final_model.ph")

model = CNNNet()
# model.load_state_dict(torch.load("./The train.EEGNet.ph"))
# file_list = os.listdir("/Users/lihao/PythonCode/model")
arr1 = []
arr2 = []
arr3 = []
arr4 = []
arr5 = []
arr6 = []
arr7 = []
arr8 = []
arr9 = []
arr10 = []

for a in range(1, 7):
    for b in range(10):
        acc = []
        test_data = CNNNetDataset(file_path='/Users/lihao/PythonCode/BCI/fNIRS/pt_testing_4D_data/sub0' + str(a) + '_test4d_0'+str(b)+'.pt',
                                  target_path='/Users/lihao/PythonCode/test_target/sub0' + str(a) + 'test_label_0'+str(b)+'.pt',
                                  transform=False, target_transform=False)
        test_dataloader = DataLoader(test_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size,
                                     drop_last=True)
        # for i in file_list:
            # model.load_state_dict(torch.load("/Users/lihao/PythonCode/model/" + i))
            # acc.append(infer(model, test_dataloader, device))
        model.load_state_dict(torch.load("/Users/lihao/PythonCode/BCI/fNIRS/model/final_model.ph"))
        if b == 0:
            arr1.append(infer(model, test_dataloader, device))
        elif b == 1:
            arr2.append(infer(model, test_dataloader, device))
        elif b == 2:
            arr3.append(infer(model, test_dataloader, device))
        elif b == 3:
            arr4.append(infer(model, test_dataloader, device))
        elif b == 4:
            arr5.append(infer(model, test_dataloader, device))
        elif b == 5:
            arr6.append(infer(model, test_dataloader, device))
        elif b == 6:
            arr7.append(infer(model, test_dataloader, device))
        elif b == 7:
            arr8.append(infer(model, test_dataloader, device))
        elif b == 8:
            arr9.append(infer(model, test_dataloader, device))
        elif b == 9:
            arr10.append(infer(model, test_dataloader, device))

acc1 = sum(arr1)/len(arr1)
acc2 = sum(arr2)/len(arr2)
acc3 = sum(arr3)/len(arr3)
acc4 = sum(arr4)/len(arr4)
acc5 = sum(arr5)/len(arr5)
acc6 = sum(arr6)/len(arr6)
acc7 = sum(arr7)/len(arr7)
acc8 = sum(arr8)/len(arr8)
acc9 = sum(arr9)/len(arr9)
acc10 = sum(arr10)/len(arr10)

import matplotlib.pyplot as plt

names = ['0', '', '2', '', '4', '','6','','8','','10']
x = range(len(names))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

y_1 = [0,acc1*100, acc2*100, acc3*100, acc4*100, acc5*100, acc6*100,acc7*100,acc8*100,acc9*100,acc10*100]
#y_2 = [3, 4, 5, 6, 1, 2]
#y_3 = [4, 5, 6, 1, 2, 3]

plt.plot(x, y_1, color='Green', marker='o', linestyle='-', label='FSNet')
#plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='B')
#plt.plot(x, y_3, color='green', marker='*', linestyle=':', label='C')
plt.legend()  # 显示图例
plt.xticks(x, names)
plt.xlabel("Time(s)")  # X轴标签
plt.ylabel("Accuracy(%)")  # Y轴标签
plt.show()

print(acc1)
print(acc2)
print(acc3)
print(acc4)
print(acc5)
print(acc6)
print(acc7)
print(acc8)
print(acc9)
print(acc10)
print(arr1)
print(arr5)