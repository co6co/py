#-*- coding: utf-8 -*-
# 地区竞争力指标姜维 （主要成分分析方法）
# 衡量我国各省市综合发展情况的一些数据，数据来源于《中国统计年鉴》。
# 选取6个指标，分别是人均GDP、固定资产投资、社会消费品零售总额、农村人均纯输入等
# 利用因此分析来提取公共因子，分析衡量发展因素指标。

import  torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import transformer

# 1. 读取数据
# 使用的模拟数据
region =pd.read_csv("./data/008.2.region.csv")
region_d=region[['x1','x2','x3','x4']] #选取 指定的四列
region['target']=region['y']

#2. 变量特征降维
# 这段代码用于将数据转换为适合特定深度学习框架使用的格式，
# 为进一步的处理和分析做好准备
transformer_1=PCA(n_components=2) # 创建一个主成分分析（PCA）对象，指定保留的主成分数量为2
region_d=transformer_1.fit_transform(region_d) # 使用拟合和转换方法对 region_d 进行主成分分析
x=torch.from_numpy(region_d) #将 numpy数值转换为 tensor
y=torch.from_numpy(np.array(region['target'])) #将 numpy数值转换为 tensor
x,y=Variable(x),Variable(y) # 将 tensor 转换为 Variable 类型，以便在计算图中进行自动求导

# 3 设置网络结构
# Sequential 按顺序堆叠神经网络各层
# 每个层都是通过Linear来定义线性变换的
#在每个线性层之后十元ReLU激活函数了引入非线性层

# 这种神经网络结构可用于多种任务（分类、回归等），具体的应用和训练过程取决与数据和任务的要求
# 实际应用中，还需要进行数据加载、损失函数定义、优化器选择和训练循环等来训练和使用这个神经网络。
net=torch.nn.Sequential(
    torch.nn.Linear(2,10),# 第一线性层，输入维度2，输出维度10
    torch.nn.ReLU(),#激活函数
    torch.nn.Linear(10,3), # 第二个线性层，输入维度10，输出3 
)
print(net) #打印神经网络结构

# 5. 设置优化器，随机梯度下降
optimizer=torch.optim.SGD(net.parameters(),lr=0.00001) # 学习率0.01
# 6. 定义损失函数
loss_func=torch.nn.CrossEntropyLoss() # 交叉熵损失函数

for t in range(100):
    out=net(x.float())
    loss=loss_func(out,y.long()) #计算损失
    optimizer.zero_grad() # 清空之前的梯度
    loss.backward() # 反向传播，计算当前梯度
    optimizer.step() # 更新参数
    if 5 % 25==0:
        plt.cla() # 情况图形
        prediction=torch.max(out,1)[1] # 获取预测结果
        pred_y=prediction.data.numpy() # 将预测结果转换为 numpy 数组
        target_y=y.data.numpy() # 将目标标签转换为 numpy 数组
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=5,cmap='coolwarm')
        accuracy=float((pred_y==target_y).astype(int).sum())/float(target_y.size)

        print('准确率Accuracy=%.2f' %accuracy)
        plt.pause(0.1)
    plt.show()

# 7. 保存网络及参数
torch.save(net,'./data/008.2.pk1')
torch.save(net.state_dict(),'./data/008.2.params.pk2')
