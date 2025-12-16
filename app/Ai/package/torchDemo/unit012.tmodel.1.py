#-*- coding: utf-8 -*-
# 未做测试
# 手写数字识别 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from visdom import Visdom

batch_size=200
learning_rate=0.01
epochs=20

# 1. 导入训练数据集 
train_loader=DataLoader(
    torchvision.datasets.MNIST(root='./data/tmp',train=True,download=False,transform=transforms.ToTensor()),
    batch_size=batch_size,shuffle=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ])
    )
test_loader=DataLoader(
    torchvision.datasets.MNIST(root='./data/tmp',train=False,download=False,transform=transforms.ToTensor()),
    batch_size=batch_size,shuffle=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ])
    )

# 搭建网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.mode1=nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,200),
            nn.LeakyReLU(inplace=True),
        )
        
        
    def forward(self,x):
        x=self.mode1(x)
        return x

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=MLP().to(device) 
# 优化器
optimizer=optim.SGD(net.parameters(),lr=learning_rate)
loss_func=nn.CrossEntropyLoss().to(device) # 交叉熵损失函数

vis=Visdom()
vis.line([0.],[0.],win='train_loss',opts=dict(title='train loss训练损失'))
vis.line([0.],[0.],win='test',opts=dict(title='test loss测试损失'))
global_step=0
for epoch in range(epochs):
    for step,(inputs,labels) in enumerate(train_loader):
        inputs=inputs.view(-1,28*28)
        data,target=inputs.to(device),labels.to(device)
        logits=net(data) 
        loss=loss_func(logits,target) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        vis.line([loss.item()],[global_step],win='train_loss',update='append')
        global_step+=1
        if step %100==0: 
            print(f"Epoch:{epoch},Step:{step},Loss:{loss.item()}")
    test_loss=0
    correct=0
     
    for inputs,labels in test_loader:
        inputs=inputs.view(-1,28*28)
        data,target=inputs.to(device),labels.to(device)
        logits=net(data) 
        loss=loss_func(logits,target) 
        pred=logits.argmax(dim=1)
        correct+=pred.eq(target).float().sum().item()
    vis.line([test_loss],correct/len(test_loader.dataset),[global_step],win='test',update='append')
    vis.images(data.view(-1,1,28,28),win='x')
    vis.text(str(pred.detach().cpu().numpy()),win='pred',opts=dict(title='预测'))
    test_loss /=len(test_loader.dataset)
    print(f"Test Loss:{test_loss:.4f},correct:{correct},{len(test_loader.dataset)},correct rate:{correct/len(test_loader.dataset):.4f}")
     
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
