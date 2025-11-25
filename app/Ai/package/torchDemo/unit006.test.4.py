# -*- coding: utf-8 -*-
# Leaky ReLU 激活函数
# f(x)=max(0.01x,x)

import numpy as np
import matplotlib.pyplot as plt
import torch    

def leaky_relu(x,alpha=0.01):
    """
    计算Leaky ReLU激活函数的值
    """
    return torch.maximum(alpha*x,x)
def plot_relu():
    x=torch.arange(-10,10,0.1)
    y=leaky_relu(x,1)
    plt.plot(x,y)
    plt.title("Leaky ReLU f(x)=max(0.01x,x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

if __name__=='__main__':
    plot_relu()

    
