#-*- coding: utf-8 -*-
#使用PYtorch库实现模型的调优  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
#1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1=nn.Linear(10,50)
        self.relu=nn.ReLU(  )
        self.fc2=nn.Linear(50,10)
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        return out
# 2. 定义损失函数和优化器
model=MyModel()
criterion=nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer=optim.SGD(model.parameters(),lr=0.001) # 随机梯度

# 数据加载
num_epochs=10
train_ds=TensorDataset(torch.tensor([i for i in range(100)],dtype=torch.float32),torch.tensor([i for i in range(100)],dtype=torch.float32))
train_dataloader=DataLoader( train_ds )
test_ds=TensorDataset(torch.tensor([i for i in range(10)],dtype=torch.float32),torch.tensor([i for i in range(10)],dtype=torch.float32))
test_dataloader=DataLoader(test_ds)

#进行模型训练
for epoch in range(num_epochs): 
    for inputs,labels in train_dataloader:
        optimizer.zero_grad()
        print(inputs,labels)
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

model.eval() #进行模型评估
with torch.no_grad():
    corrent=0
    total=0
    for inputs,labels in test_dataloader:
        outputs=model(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=correct/total
    print(f"Accuracy:{accuracy*100:.2f}")
