#-*- coding: utf-8 -*-
# 构建简单神经网络
# 用于手写数字识别任务
# 进行训练和测试

import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision # 计算机视觉库
import torchvision.transforms as transforms
 

# 数据预处理 
transform=transforms.Compose({
    transforms.ToTensor(), # 将图像转换为张量
    transforms.Normalize((0.5,),(0.5,)) # 标准化处理，将张量的值从 [0, 1] 映射到 [-1, 1]
    })
dataRoot='./data/tmp/007.1'
dataRoot_test='./data/tmp/007.1.test'
# 创建数据集 （指定路径，是否为训练集，是否下载，预处理）
train_set=torchvision.datasets.MNIST(root=dataRoot,train=True,download=True,transform=transform)
# 创建训练数据加载器，(批量大小、打乱顺序，工作现线程)
train_loader=DataLoader(train_set,batch_size=100,shuffle=True,num_workers=2)
test_set=torchvision.datasets.MNIST(root=dataRoot_test,train=False,download=True,transform=transform)
test_loader=DataLoader(test_set,batch_size=100,shuffle=False,num_workers=2)

# 2. 定义神经网络模型
class Net(nn.Module):
    """
    两个卷积曾、两个全连接层和一个池化层
    """
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)# 卷积1层、输入通道数为1，输出通道数为6、卷积核大小5x5
        self.pool=nn.MaxPool2d(2,2) # 最大池化层、池化核大小2x2，步长2
        self.conv2=nn.Conv2d(6,16,5) # 卷积2层，输入通道6，输出通道16，卷积核大小5x5
        self.fc1=nn.Linear(16*4*4,120) # 全连接曾1，输入节点数256，输出节点数120
        self.fc2=nn.Linear(120,84) # 全连接曾2，输入节点数120，输出节点数84
        self.fc3=nn.Linear(84,10)  # 全连接曾3，输入节点数84，输出节点数10
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) # 进行第一次卷积操作，然后应用ReLU激活函数进行最大池化操作
        x=self.pool(F.relu(self.conv2(x)))  # 第二次卷积操作
        x=x.view(-1,16*4*4) # 展平操作，将特征图展开为一维向量
        x=F.relu(self.fc1(x)) # 进行第一层全连接操作，然后应用ReLU激活函数
        x=F.relu(self.fc2(x)) # 进行第二层全连接操作，然后应用ReLU激活函数
        x=self.fc3(x) # 进行第三层全连接操作，得到最终的输出结果 
        return x

net=Net()
# 3. 定义损失函数和优化器
criterion=nn.CrossEntropyLoss() # 交叉熵损失
# 随机梯度下降优化器
# net.parameters() 将神经元草书穿过优化器，以便让优化器知道要更新那些参数
# lr 学习率 优化器的超参数，用于控制每次参数更新的步长
# momentum SGD的超参，用于加速收敛过程，会保留之前步骤的梯度，并考虑当前步骤的梯度和之前步骤的梯度的加权平均值来更新参数
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9) 

# 训练
for epoch in range(10):
    running_loss=0.0 #每轮的损失函数数值为0
    for i,data in enumerate(train_loader,0):
        input,labels=data # 获取输入数值和对应的标签
        optimizer.zero_grad() # 梯度置零,以便下一次反向传播
        outputs=net(input) # 前向传播，计算输出结果
        loss=criterion(outputs,labels) # 计算损失函数值
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 更新参数
        running_loss+=loss.item() # 累加损失函数值
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}") # 打印每轮的损失函数值

# 测试
correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        images,labels=data
        outputs=net(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy on test set: {100*correct/total}%")


