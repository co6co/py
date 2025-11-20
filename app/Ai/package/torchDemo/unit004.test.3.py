#-*- coding: utf-8 -*-
# 1. 均值、中位数、众数、方差和标准差 
# 2. 创建两个张量，用矩阵广播机制计算 加法、减法、乘法、除法


import torch
x=torch.randn(10)  # 创建一个包含10个随机数的张量
print("原始张量:", x)
mean=x.mean()
print("均值:", mean)
median=x.median()
print("中位数:", median)
mode=x.mode().values
print("众数:", mode)
variance=x.var()
print("方差:", variance)
std=x.std()
print("标准差:", std)

# 2.
# 创建两个张量，用矩阵广播机制计算 加法、减法、乘法、除法
a=torch.tensor([[1,2,3],[4,5,6]])
b=torch.tensor([[10,20,30]])
print("张量a:", a)
print("张量b:", b)
print("加法:", a+b)
print("减法:", a-b)
print("乘法:", a*b)
print("除法:", a/b)
