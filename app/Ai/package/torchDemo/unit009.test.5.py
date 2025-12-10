#-*- coding: utf-8 -*-
# 图像分割任务
# 选择适当的图像分割任务，例如物体检测、语义分割或实例分割
# 使用合适的数据集进行训练和测试
# 
from turtle import forward
import torch 
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torchvision .datasets import CIFAR10

from demo import num

# 加载
trainSet=torchvision.datasets.Cityscapes("./data/tmp/009.5",split='train',mode='fine',target_type='semantic',transform=transforms.ToTensor())
testSet=torchvision.datasets.Cityscapes("./data/tmp/009.5",split='test',mode='fine',target_type='semantic',transform=transforms.ToTensor())
trainLoader=DataLoader(trainSet,batch_size=32,shuffle=True,num_workers=2)
testLoader=DataLoader(testSet,batch_size=32,shuffle=False,num_workers=2)

# 模型
class SegmentationModel(nn.Module):
    def __init__(self,num_classes:int) -> None:
        """
        @param num_classes 目标类别数量
        """
        super(SegmentationModel,self).__init__()
        self.conv1=nn.Conv2d(3,64,3,1,1)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(64,64,3,1,1)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(64,num_classes,3,1,1) 
    def forward(self,x):
        x=self.relu1(self.conv1(x))
        x=self.relu2(self.conv2(x))
        x=self.conv3(x)
        return x

model=SegmentationModel(20) 
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

# 训练
num_epochs=20
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in rang(num_epochs):
    model.train()
    for images,labels in trainLoader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

# 测试
model.eval()
total_corrent=0
total_samples=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total_corrent+=(predicted==labels).sum().item()
        total_samples+=labels.numel()

accuracy=total_corrent/total_samples
print(f"Test Accuracy 准确率: {accuracy:.4f}")
