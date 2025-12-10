
#-*- coding: utf-8 -*- 
# 训练网络模型
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
from unit009_test_2_0 import MyDataset
from unit009_test_2_1 import MyDataset,Net
# 批次大小
batchsize=8
# 训练轮次
epochs=20
# 数据路径
train_data_path='./data/009.2/train'
# 数据预处理
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
# 构建自定义数据集
bag=MyDataset(train_data_path,transform)
# 数据加载器
dataloader=DataLoader(bag,batch_size=batchsize,shuffle=True)
# 选择设备
device=torch.device('cpu')
# 神经网络
net=Net()
net.to(device)
#损失函数
criterion=nn.BCELoss()
# 优化器
optimizer=optim.Adam(net.parameters(),lr=le-2,momentum=0.7)

for epoch in range(1,epochs+1):
    
    for batch_idx,( img,lab) in enumerate(dataloader):
        img,lab=img.to(device),lab.to(device)
        output=torch.sigmoid(net(img))
        loss=criterion(output,lab)
        output_np=output.cpu().data.numpy().copy() # 输出转NumPy格式并复制
        output_np=np.argmin(output_np,axis=1) # 取最小概率的类别索引
        y_np=lab.cpu().data.numpy().copy()
        y_np=np.argmin(y_np,axis=1) # 取最大概率的类别索引
        if batch_idx%10==0: 
            print(f'Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Acc: {acc:.4f}')
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if batch_idx%10==0:
            torch.save(net.state_dict(),f'./model/009.2/unit009.test.2.epoch{epoch}.pth') 

# 应用模型
class TestDataSet(torch.utils.data.Dataset):
    def __init__(self,test_path,transform=None):
        self.test_img=os.listdir(test_path)
        self.transform=transform
        self.images=[] # 存储处理后的图片路径 
        for i in range(len(self.test_img)):
            self.images.append(os.path.join(test_path,self.test_img[i]))
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img_path=self.images[index]
        img=cv2.imread(img_path)
        img=cv2.resize(img,(224,224))
        if self.transform is not None:
            img=self.transform(img) 
        return img

test_img_path='./data/tmp/009_2/test/last'
checkpoint_path='./data/tmp/model_epoch20.pth'
save_dir='data/tmp/009_2/test/result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 构建变换
bag=TestDataSet(test_img_path,transform)
# 数据加载器
dataloader=DataLoader(bag,batch_size=batchsize,shuffle=False)
net=torch.load(checkpoint_path) # 加载检查点模型
for idx,img in enumerate(dataloader):
    output=torch.sigmoid(net(img))
    output_np=output.cpu().data.numpy().copy() # 输出转NumPy格式并复制
    output_np=np.argmin(output_np,axis=1) # 取最小概率的类别索引 
    # 压缩输出并乘以255
    img_arr=np.squeeze(output_np)
    img_arr=img_arr.astype('uint8')*255
    #保存图像
    cv2.imwrite(f"{save_dir}/{idx}",img_arr)
    print(f"{save_dir}/{idx}")