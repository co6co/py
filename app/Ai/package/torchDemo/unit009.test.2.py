#-*- coding: utf-8 -*-
# 未做测试
# 自动分割模型 
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

#  加载数据集的类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train_path,transform=None):
        self.images=os.listdir(train_path+'/last')
        self.labels=os.listdir(train_path+'/label_msk')
        assert len(self.images)==len(self.labels)

        self.transform=transform
        self.images_and_labels=[] # 存储图像路径和标签路径列表
        for i in range(len(self.images)):
            self.images_and_labels.append((train_path+'/last/'+self.images[i],train_path+'/label_msk/'+self.labels[i]))
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img_path,lab_path=self.images_and_labels[index]
        img=cv2.imread(img_path)
        img=cv2.resize(img,(224,224))
        lab=cv2.imread(lab_path,0)
        lab=cv2.resize(lab,(224,224))
        lab=lab/255
        lab=np.eye(2)[lab]
        # 对one-hot 编码进行处理
        lab=lab.transpose(0,2,1)
        if self.transform is not None:
            img=self.transform(img) 
        return img,lab


if __name__=='__main__':
    
    img=cv2.imread('/data/'+'/last/50.jpg',0)
    img=cv2.resize(img,(16,16))
    img2=img/255
    img3=img2.astype('uint8')
    hot1=np.eye(2)[img3] # 对标签进行one-hot 编码
    hot2=np.array(list(map(lambda x: abs(x-1),hot1))) 
    print(hot2.shape)