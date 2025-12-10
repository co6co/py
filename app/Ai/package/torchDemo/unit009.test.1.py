#-*- coding: utf-8 -*-
# 未做测试
# 手写数字识别 
from functools import lru_cache
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 1. 导入训练数据集
train_data=torchvision.datasets.MNIST(root='./data/tmp',train=True,download=False,transform=transforms.ToTensor())
test_data=torchvision.datasets.MNIST(root='./data/tmp',train=False,download=False,transform=transforms.ToTensor())
# 将测试数据的数据部分增加一个维度，并将数据类型转换为浮点数张量然后除以255
test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)/255.0
# 测试数据的目标
test_y=test_data.targets

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)

# 搭建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1), # 二维卷积层 输入通道1，输出通道16，卷积核3x3，步长1，填充1
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 最大池化层 核大小 2*2
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=2), # 二维卷积层 输入通道16，输出通道32，卷积核3x3，步长1，填充2
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 最大池化层 核大小 2*2
        )
        self.output=nn.Linear(32*7*7,10) # 全连接层 输入32*7*7，输出10
    def forward(self,x):
        x=self.conv1(x) #对输入数据进行第一层卷积和池化
        x=self.conv2(x) #
        x=x.view(out.size(0),-1) # 输出展平为一维
        x=self.output(x)  
        return x
cnn=CNN()

# 优化器
optimizer=optim.Adam(cnn.parameters(),lr=0.001)
loss_func=nn.CrossEntropyLoss() # 交叉熵损失函数
for epoch in range(10):
    for step,(inputs,labels) in enumerate(train_loader):
        output=cnn(inputs)
        outputs=cnn(inputs)
        loss=loss_func(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%100==0:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,dim=1)[1].data.numpy()
            accuracy=float((pred_y==test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
            print(f"Epoch:{epoch},Step:{step},Loss:{loss.item()},Accuracy:{accuracy}")
torch.save(cnn.state_dict(),f"./data/009.1.cnn_minist.pth")

# 对数据集进行预测
cnn=torch.load(f"./data/009.1.cnn_minist.pth")
test_output=cnn(test_x[:20])
pred_y=torch.max(test_output,1)[1].data.numpy()
print("预测值",pred_y)
print('实际值',test_y[:20].numpy())
test_output1=cnn(test_x)
pred_y1=torch.max(test_output1,1)[1].data.numpy()
# 计算准确率
accuracy=float((pred_y1==test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('准确率',accuracy) 
