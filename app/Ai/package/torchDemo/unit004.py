#-*- coding: utf-8 -*-
# 以概率论为基础，研究随机现象的统计规律
# 数理统计
# 从数量化的角度来研究现实世界中的随机现象
import torch
x=torch.tensor([[1,2],[3,4]]) 
## 1. 求积
# dim 参数默认为None，给定将是一个降维度的张量
# keepdim：是否保持原始维度，默认False
result=torch.prod(x) # 求积
print("x的积:",result)
result=torch.prod(x,dim=0) # 按列求积 tensor([3, 8])
print("torch.prod(x,dim=0) x按列求积:",result)
result=torch.prod(x,dim=1) # 按行求积 tensor([2, 12])
print("torch.prod(x,dim=1) x按行求积:",result)

# 求和
result=torch.sum(x ) # 求和
print("x的和:",result)
result=torch.sum(x,dim=0) # 按列求和 tensor([4, 6])
print("torch.sum(x,dim=0) x按列求和:",result)
result=torch.sum(x,dim=1,keepdim=True) # 按行求和 tensor([3, 7])
print("torch.sum(x,dim=1) x按行求和:",result)
result=torch.sum(x,dim=(0,1),keepdim=True) # 按行求和 tensor([3, 7])
print("torch.sum(x,dim=(0,1),keepdim=True) x多维求和:",result)
# 平均值、最大值、最小值
result=torch.mean(x,dtype=torch.float32) # 平均值
print("x的平均值:",result) 
result=torch.max(x) # 最大值
print("x的最大值:",result) 
result=torch.min(x) # 最小值
print("x的最小值:",result) 
# 中位数、众数
result=torch.median(x) # 中位数
print("x的中位数:",result) 
result=torch.mode(x,1) # 众数
print("x的按行的众数:",result) 

# 标准差、方差
x=torch.tensor([[1,2],[3,4]],dtype=torch.float32)
result=torch.std(x ) # 标准差
print("x的标准差:",result) 
result=torch.var(x  ) # 方差
print("x的方差:",result) 
result=torch.var(x,unbiased=False) # 无偏方差
print("使用不使用无偏估算方差:",result) 

# 取整
x=torch.tensor(3.1415926)
result=torch.floor(x) # 向下取整
print("torch.floor(x) 向下取整:",result)
result=torch.ceil(x) # 向上取整
print("torch.ceil(x) 向上取整:",result)
result=torch.round(x) # 四舍五入
print("torch.round(x) 四舍五入:",result)

result=torch.trunc(x) # 截断取整
print("torch.trunc(x) 截断取整:",result)
result=torch.frac(x) # 取小数部分
print("torch.frac(x) 取小数部分:",result)