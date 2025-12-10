#-*- coding: utf-8 -*-
# 使用 torch库实现主成分分析
# 首先对原始数据进行中心化，然后计算协方差矩阵，并对协方差进行特征分解，根据降维后的维度k
#选择k个特征向量作为主成分，最后中心化后的数据乘以主成分矩阵，得到降维后的数据
# 实际应用中可能需要进行更复杂的数据处理和参数调整。
import torch
from torch import tensor
from torch.linalg import svd

# 假设有一组数据X，每行代表一个样本，每列代表一个特征
X=tensor([[1,2],
        [3,4],
        [5,6]],dtype=torch.float32)
# 对数据进行中心化
X_centered=X-X.mean(dim=0)
#计算数据的协方差矩阵
cov_matrix=torch.matmul(X_centered.T,X_centered)/(X_centered.shape[0]-1)
# 对协方差矩阵进行特征值分解
_,_,V=svd(cov_matrix)
# 选择前k个特征向量作为主成分
k=1 
principal_components=V[:,:k]
# 将数据投影到主成分上
X_projected=torch.matmul(X_centered,principal_components)
print(X_projected)
