#-*- coding: utf-8 -*-
# 图像识别任务
# 通常使用 torchvision.dataset.ImageFolder或自定义 torch.utils.data.Dataset类 库来加载
# 使用 torchvision.transforms 对图像进行预处理 调整图像大学 灰度、归一化
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision .datasets import CIFAR10

from unit009 import criterion

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5,1,2)
        self.relu=nn.ReLU()
        self.maxpoll=nn.MaxPool2d(2,2)
        
        self.fc=nn.Linear(16*14*14,10) 
    def forward(self,x): 
        x=self.maxpoll(self.relu(self.conv1(x))) 
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

#加载数据集 
trainSet=CIFAR10(root='./data/tmp/009.4',train=True,download=True,transform=transforms.ToTensor())
trainLoader=DataLoader(trainSet,batch_size=64,shuffle=True,num_workers=2)
TestSet=CIFAR10(root='./data/tmp/009.4',train=False,download=True,transform=transforms.ToTensor())
testLoader=DataLoader(TestSet,batch_size=64,shuffle=False,num_workers=2)

# 定义模型 随时函数和优化器
model=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 训练
for epoch in range(10):
    running_loss=0.0
    for i,(data,target) in enumerate(trainLoader,0):
        data,target=data.to(device),target.to(device)
        outputs=model(data)
        loss=criterion(outputs,target)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        running_loss+=loss.item()
        if i%100==199:
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/200))      
            running_loss=0.0

# 测试模型
model.eval()
with torch.no_grad():
    correct=0
    total=0
    for data,targets in testLoader:
        data,targets=data.to(device),targets.to(device)

        output=model(data)
        _,predicted=torch.max(output.data,1)
        total+=targets.size(0)
        correct+=(predicted==targets).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
