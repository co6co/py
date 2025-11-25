#-*- coding: utf-8 -*-
# sigmoid 激活函数
#f(x)=1/(1+e^(-x))
import torch
import matplotlib.pyplot as plt
import numpy as np
# 定义sigmoid 函数
def sigmoid(x):
    return 1/(1+torch.exp(-x)) #np.exp(-x)
# 用于绘制sigmoid函数图像
def plot_sigmoid():
    #x=np.arange(-10,10,0.1) 
    #x=torch.from_numpy(x)
    x=torch.range(-10,10,0.1) #生产x轴数值，范围 -10~10，步长0.1
    y=sigmoid(torch.from_numpy(x)) # 计算y轴数据，即sigmoid值

    # 绘制图像
    plt.plot(x,y.numpy())
    plt.title("Sigmoid f(x)=1/(1+e^(-x))")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

if __name__ == "__main__":
    plot_sigmoid()