# -*- coding: utf-8 -*-
# L1 损失函数

from pickle import FALSE
import torch
# 1. L1损失函数

# 定义一个L1损失函数，计算输入和目标之间的平均绝对误差
loss=torch.nn.L1Loss(reduction='sum') # none,mean,sum
input=torch.tensor([1.0,2.0,3.0,4.0])
target=torch.tensor([4.0,5.0,6.0,7.0])
# 计算L1损失
output=loss(input,target)
print(output)

# 2. MSELoss 均方误差损失函数
loss=torch.nn.MSELoss(reduce=True,size_average=False,reduction='mean')
input=torch.tensor([1.0,2.0,3.0,4.0])
target=torch.tensor([4.0,5.0,6.0,7.0])
# 计算MSE损失
output=loss(input,target)
print(output)
