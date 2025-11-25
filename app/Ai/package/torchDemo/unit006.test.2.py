#-*- coding: utf-8 -*-
# Tanhs 激活函数
# 正切和余切的推到函数
# f(x)=(e^x-e^(-x))/(e^x+e^(-x))
 
import matplotlib.pyplot as plt
import torch

def tanh(x):
    """
    计算Tanh激活函数的值
    """
    #return np.exp(x)-np.exp(-x)/(np.exp(x)+np.exp(-x))
    return (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
# 用于绘制Tanh函数图像
def plot_tanh(): 
    x=torch.range(-10,10,0.1) #生产x轴数值，范围 -10~10，步长0.1
    y=tanh(x) # 计算y轴数据，即Tanh值

    # 绘制图像
    plt.plot(x,y )
    plt.title("Tanh f(x)=(e^x-e^(-x))/(e^x+e^(-x))")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
 
if __name__ == "__main__":
    plot_tanh()