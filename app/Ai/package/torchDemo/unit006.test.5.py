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

# 3. 交叉熵损失函数
entroy_loss=torch.nn.CrossEntropyLoss() # 交叉熵损失
input=torch.tensor([[0.1,0.2,0.3]])
target=torch.tensor([0])
output=entroy_loss(input,target)
print("交叉熵损失:",output)

# 4. 余弦相似度损失函数
# 定义输入向量x1,x2,以及目标标签y,其中y=1表示x1和x2相似，y=-1表示x1和x2不相似
x1=torch.tensor([1.0,2.0,3.0,4.0])
x2=torch.tensor([4.1,6.1,7.1,8.1])

similarity=torch.cosine_similarity(x1,x2,dim=0)# 计算两个张量的余弦相似度，dim=0 表示按列计算
loss=1-similarity # 计算余弦相似度损失，1-余弦相似度
# 计算余弦相似度损失
print("余弦相似度损失:",loss)

x1=torch.tensor([1.0,2.0,3.0])
x2=torch.tensor([4.0,5.0,6.0])
y=torch.tensor([1,-1])
# 计算余弦相似度损失
loss=torch.nn.CosineEmbeddingLoss(margin=0.5)
output=loss(x1,x2,y)
print("余弦相似度损失2:",output) # tensor(0.5000)
