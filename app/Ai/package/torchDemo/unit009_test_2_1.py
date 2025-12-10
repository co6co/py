
#-*- coding: utf-8 -*- 
# 自动分割模型  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset 
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 

# 神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 第一层编码块
        self.encode1=nn.Sequential (
            nn.Conv2d(3,64,3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        # 第二层编码块
        self.encode2=nn.Sequential (
            nn.Conv2d(64,128,3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
         # 第三层编码块
        self.encode3=nn.Sequential (
            nn.Conv2d(128,256,3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
         # 第四层编码块
        self.encode4=nn.Sequential (
            nn.Conv2d(256,512,3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
         # 第五层编码块
        self.encode5=nn.Sequential (
            nn.Conv2d(512,512,3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        # 定义第一层解码块，有卷积转置层、归一化层和激活函数组成
        self.decode1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decode2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decode3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decode4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decode5=nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        # 定义分类器，用于生产最终的输出
        self.classfier=nn.Conv2d(16,2,kernel_size=1)
    def forward(self,x):
        out=self.encode1(x)
        out=self.encode2(out)
        out=self.encode3(out)
        out=self.encode4(out)
        out=self.encode5(out)
        out=self.decode1(out)
        out=self.decode2(out)
        out=self.decode3(out)
        out=self.decode4(out)
        out=self.decode5(out)
        out=self.classfier(out)
        return out
        
         
if __name__=='__main__':
    # 随机的图像张量
    img=torch.randn(2,3,244,244)
    net=Net()
    sample=net(img) # 对随机图进行向前传播
    print(sample.shape)
     