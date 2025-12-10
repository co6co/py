#-*- coding: utf-8 -*-
#创建图像自动分类器 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的GPU设备
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

# 加载数据集
transform=transforms.Compose([
    transforms.ToTensor(), # 数据转换为张量格式
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 对数据进行归一化处理，其均值为0.5，标准差为0.5
])
# 训练集
trainset=torchvision.datasets.CIFAR10(root='./data/tmp',train=True,download=True,transform=transform)
trainloader=DataLoader(trainset,batch_size=4,shuffle=False,num_workers=2)
# 测试集
testset=torchvision.datasets.CIFAR10(root='./data/tmp',train=False,download=True,transform=transform)
testloader=DataLoader(testset,batch_size=4,shuffle=True,num_workers=4)
# 定义列别名称
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) # 卷积层
        self.pool=nn.MaxPool2d(2,2) # 池灰层 池化窗口2*2
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) # 对输入数据进行卷积和激活操作，并进行池化
        x=self.pool(F.relu(self.conv2(x))) # 对池化后的数据进行卷积核激活操作
        x=x.view(-1,16*5*5) #将数据展平为一维
        x=F.relu(self.fc1(x)) # 对战平后的数据进行全连接和激活操作
        x=F.relu(self.fc2(x)) # 对上一层的输出进行全连接操作
        x=self.fc3(x) # 返回最终输出
        return x

net=Net()

# 设置损失函数和优化器
criterion=nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9) # 随机梯度下降优化器

# 训练网络模型
num_epochs=2
for epoch in range(num_epochs):
    _loss=0.0
    for i,(inputs,labels) in enumerate(trainloader,0): 
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _loss+=loss.item()
        if i%3000==3999:
            print(f"[{epoch+1},{i+1}] loss:{_loss/2000:.3f}")
            _loss=0.0
print("Finished Training")

# 应用网络模型
def imshow(img:torch.Tensor):
    """显示图像
    img 要显示的图像
    """
    img=img/2+0.5 # 对图像进行反归一化处理
    npimg=img.numpy()# 将Tensor转换为Numpy数组
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# 迭代测试集加载器
dataiter=iter(testloader)
inputs,labels=next(dataiter)
imshow(torchvision.utils.make_grid(inputs)) # 使用make_grid 函数将图像拼接成网格形式
print("图片真是分类:",' '.join(f"{classes[labels[j]]}" for j in range(4)))

outputs=net(inputs)
_,predicted=torch.max(outputs,1)
print("图片预测分类:",' '.join(f"{classes[predicted[j]]}" for j in range(4)))

# 计算测试集准确率
correct,total=0,0 # 正确预测的数量和总预测数量
with torch.no_grad(): # 禁用梯度计算，以进行测试
    for images,labels in testloader:
        outputs=net(images)
        _,predicted=torch.max(outputs,1) # 在输出的第一维（每个图像）上找到最大值及其索引，得到预测的类别predicted
        total+=labels.size(0)#累计总预测数量total，通过 labels.size(0) 获取标签的数量
        correct+=(predicted==labels).sum().item() # 累计正确预测的数量correct，通过比较标签和预测结果，使用sum().item() 将张量转换为整数
accuracy=correct/total
print(f"Accuracy 测试集准确率:{accuracy*100:.2f}")