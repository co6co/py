#-*- coding: utf-8 -*-
# 聚类分析
# 
# 植物花卉特征聚类
#根据花瓣长度、宽度、花萼长度、宽度4个特征进行聚类分析。
#数据集包含3类共150条记录，每类各50条记录。
# 使用kmeans_pytorch 包中的K-Means算法实现聚类分析

import torch
import numpy as np
import pandas as pd
from kmeans_pytorch import kmeans # pip install kmeans-pytorch
import matplotlib.pyplot as plt 

# 设置运行环境
if torch.cuda.is_available(): # 是否有可用的 CUDA设备
    device = torch.device('cuda:0') # 使用CUDA设备0
else:
    device = torch.device('cpu') # 使用CPU 
# 加载数据
#from sklearn.datasets import load_iris
#import pandas as pd
#
#iris = load_iris() # 加载 iris.zip 格式的数据
#X = iris.data  # 特征数据 (150, 4)
#y = iris.target  # 目标标签 (150,)
#
## 转换为DataFrame
#df = pd.DataFrame(X, columns=iris.feature_names)
#df['target'] = y
#df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# 1. 读取数据
dataPath='data/008-1iris.csv'
plant = pd.read_csv(dataPath)
# 选择需要的列
plant_d=plant[['sepal_length','sepal_width','petal_length','petal_width']]
# 将Species字符串标签转换为数字标签
plant['target'], class_names = pd.factorize(plant['species'])
print("类别名称:", class_names)
print("数字标签:", plant['target'])
# 将数据转化Numpy数组，然后通过torch.from_numpy()函数将其转换为Torch张量
x=torch.from_numpy(np.array(plant_d))
y=torch.from_numpy(np.array(plant['target']))

# 2. 设置聚类模型
num_clusters=3 # 聚类数
# 聚类模型
cluster_ids_x, cluster_centers = kmeans(
    X=x,  #输入数据
    num_clusters=num_clusters, #聚类数
     distance='euclidean',  # 距离度量方式，这里使用欧氏距离
     device=device # 设备
)
print("聚类ID->",cluster_ids_x)  
print("聚类中心点->",cluster_centers) 

# 绘制聚类后的散点图
plt.figure(figsize=(4,3),dpi=160)
#在图形上绘制散点图
#x轴和y轴的数据分别为x距离的第0列和第1列，颜色根据Cluster_ids_x进行映射
# 使用‘cool’颜色映射，标记为 D
plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool',marker='D')
# 在图形中绘制聚类中心点的散点图，
# x轴和y轴分别为Cluster_centers矩阵的第0列和第1列，透明度为0.5，
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1], 
    c='white',
    alpha=0.5,
    edgecolors='black',
     linewidths=2,
)
plt.tight_layout() # 自动调整图形布局，使图形元素适当地排列
plt.show() 



