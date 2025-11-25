# -*- coding: utf-8 -*-
# ReLU 激活函数
# f(x)=max(0,x)
 
import matplotlib.pyplot as plt
import torch

def relu(x):
    """
    计算ReLU激活函数的值
    """
    print(torch.zeros_like(x))
    return torch.maximum(torch.zeros_like(x),x)

def plot_relu():
    x=torch.range(-10,10,0.1)
    y=relu(x)
    # 绘制图像
    plt.plot(x,y)
    plt.title("ReLU f(x)=max(0,x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    
if __name__ == "__main__":
    plot_relu()
