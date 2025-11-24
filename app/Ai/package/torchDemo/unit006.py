#-*- coding: utf-8 -*-
# torch 基础
# 1. 使用 mean 与 std 创建 正态分布的tensor
import torch
mean=torch.arange(1,10,dtype=torch.float32)
std=torch.arange(1,10,dtype=torch.float32)
t=torch.normal(mean,std) #正态分布
print(f"mean:{mean}\nstd:{std}\n{t}")

# 2. 从标准正态分布，抽取随机数创建
t=torch.randn(3,3 )
print(f"t:\n{t}")

t2=torch.randn_like(t,dtype=torch.float16)
print(f"t2:\n{t2}")
 
