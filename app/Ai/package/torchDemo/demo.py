import torch
import torch.nn as nn # 用于构建神经网络模型
from torch.utils.data import DataLoader #DataLoader 、TensorDataset 用于加载数据
from torch.utils.data import TensorDataset 
import numpy as np
import matplotlib # 用于绘制图形
import matplotlib.pyplot as plt # 用于绘制图形
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
matplotlib.rcParams['axes.unicode_minus'] = False # 用于显示负号，避免出现乱码
num=400
x=np.linspace(-2*np.pi,2*np.pi,num) # 生成-2π 到 2π 的 400 个点
y=np.cos(x) # 计算余弦函数值    
X=np.expand_dims(x,axis=1) # x扩展一维数组X，即在列方向上增加一个维度
Y=y.reshape(num,-1) #  y 整形为 400行1列的矩阵Y
Y2=np.expand_dims(y,axis=1)
print("x",x)
print("X",X)
print("y",y)
print("Y",Y)
print("Y2",Y2)
