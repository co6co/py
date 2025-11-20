#-*- coding: utf-8 -*-
# 自动微分
# 自动 微分 通过 autograd实现，
# 当我们创建一个张量时，可以设置 requires_grad=True ，Pytorch 就会自动追踪该张量上的所有操作
# 当我们调用 .backward() 方法时，Pytorch 会自动计算所有相关张量的梯度，并将其存储在 .grad 属性中

# 请使用自动微分计算函数 f(x)=x^2+2x+1 在 x=3 处的导数
import torch
x=torch.tensor([3.0],requires_grad=True) # 创建张量，设置 requires_grad=True以跟踪其梯度
f=x**2+2*x+1 # 定义函数
f.backward() #使用自动微分 计算y关于x的导数
print("导数值", x.grad.item()) # 输出x的导数
