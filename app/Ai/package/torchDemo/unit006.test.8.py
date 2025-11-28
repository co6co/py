#-*- coding: utf-8 -*-
# 损失函数的作用
import torch
import torch.nn as nn
# 创建随机张量

predictions=torch.randn([10,10])  # 10 个有样本，每个样本有10个类别
targets=torch.randint(0,10,(10,))  # 10 个样本的目标类别，范围0~9，假设是分类任务
# 均方差损失
mse_loss=nn.MSELoss() 
print("MSE Loss:",mse_loss(predictions,targets))

# 计算交叉熵损失
cross_entropy_loss=nn.CrossEntropyLoss()
print("Cross Entropy Loss:",cross_entropy_loss(predictions,targets))

# 计算余弦相似度损失（需要一个额外的目标向量）
cosine_loss=nn.CosineSimilarity(dim=1,eps=1e-6) 
print("Cosine Similarity Loss:",cosine_loss(predictions,targets))

# 计算L1 损失
l1_loss=nn.L1Loss()
print("L1 Loss:",l1_loss(predictions,targets))
