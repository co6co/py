#-*- coding: utf-8 -*-
# 实现图像分类任务
from unittest import TestSuite
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F



import torchvision.transforms as transforms
from torchvision .datasets import CIFAR10

# 数据预处理
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainSet=CIFAR10(root='./data/tmp/009.3',train=True,download=True,transform=transform)
trainLoader=DataLoader(trainSet,batch_size=64,shuffle=True,num_workers=2)
TestSet=CIFAR10(root='./data/tmp/009.3',train=False,download=True,transform=transform)
testLoader=DataLoader(TestSet,batch_size=64,shuffle=False,num_workers=2)
# 定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
net=Net()

# 优化器损失函数
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# 训练
for epoch in range(10):
    running_loss=0.0
    for i,data in enumerate(trainLoader,0):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/200))
            running_loss=0.0
    # 测试
    correct=0
    total=0
    with torch.no_grad():
        for data in testLoader:
            images,labels=data
            outputs=net(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))