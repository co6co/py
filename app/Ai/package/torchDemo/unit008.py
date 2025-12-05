#-*- coding: utf-8 -*-
# 数据建模
# 住房价格回归预测 -- 回归分析
# \\todo 代码未能正常运行还需调试
#
# 影响住房价格因素
# Id    住房编号
# Area  住房面积
# Shape  户型
# Style  房屋类型
# Utilities 配套设施如通不通水电气
# Neighborhood  地理位置
# Price  住房价格

from numpy.core import numeric
import torch
import numpy as np
import pandas as pd
from torch.utils.data import  DataLoader,TensorDataset
import time
 

# 1. 准备数据
start = time.perf_counter()

# 读取训练数据和预测数据：
o_train=pd.read_csv('data/008_train.csv')
o_test=pd.read_csv('data/008_test.csv')

# 合并数据集 
all_features=pd.concat((o_train.loc[:,'Area':'Neighborhood'],o_test.loc[:,'Area':'Neighborhood']))
all_labels=pd.concat((o_train.loc[:,'Price'],o_test.loc[:,'Price']))

# 2. 数据预处理
# 提取所有特征中数值类型的特征索引
numeric_feats=all_features.dtypes[all_features.dtypes!='object'].index
# 提取所有特征中对象类型的特征索引
object_feats=all_features.dtypes[all_features.dtypes=='object'].index
# 对数值类型的特征进行标准化处理
all_features[numeric_feats]=all_features[numeric_feats].apply(lambda x:(x-x.mean())/x.std())
# 对对象类型的特征进行独热编码处理
all_features=pd.get_dummies(all_features,prefix=object_feats,dummy_na=True)
all_features=all_features.fillna(all_features.mean())

# 3. 分离训练集和测试集 
#train_features=torch.from_numpy(train_features) # 将NumPy数组转化为 Torch张量
#train_labels=torch.from_numpy(train_labels).unsqueeze(1) # 在第一个维度上对张量进行unsqueeze操作，增加一个维度
#test_features=torch.from_numpy(test_features)
#test_labels=torch.from_numpy(test_labels).unsqueeze(1)
n_train = o_train.shape[0]
train_features = all_features[:n_train].values
test_features = all_features[n_train:].values
train_labels = o_train['Price'].values
test_labels = o_test['Price'].values

# 保存均值和标准差用于后续预测
mean = o_train['Price'].mean()
std = o_train['Price'].std()

# 4. 数据类型转换，数组转换成张量
train_features=torch.from_numpy(train_features.astype(np.float32)) # 将NumPy数组转化为 Torch张量
train_labels=torch.from_numpy(train_labels.astype(np.float32)).unsqueeze(1) # 在第一个维度上对张量进行unsqueeze操作，增加一个维度
test_features=torch.from_numpy(test_features.astype(np.float32))
test_labels=torch.from_numpy(test_labels.astype(np.float32)).unsqueeze(1)
train_set=TensorDataset(train_features,train_labels) # 将训练特征和标签打包成一个数据集
test_set=TensorDataset(test_features,test_labels) # 将测试特征和标签打包成一个数据集

# 4 设置数据迭代器
batch_size=64
train_data=DataLoader(train_set,batch_size,shuffle=True) # 对训练数据集进行随机批量迭代
test_data=DataLoader(test_set,batch_size,shuffle=False) # 对测试数据集进行批量迭代

# 5. 设置网络结构
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super().__init__()
        self.layer1=torch.nn.Linear(n_feature,600) # 全连接层，输入特征 n_feature,输出维度600
        self.layer2=torch.nn.Linear(600,1200) # 全连接层2，输入维度600，输出温度1200
        self.layer3=torch.nn.Linear(1200,n_output) # 全连接层3，输入维度1200，输出维度n_output


    def forward(self,x):
        y_pred=self.layer1(x)
        y_pred=torch.relu(y_pred) # 对layer1的输出应用ReLU激活函数
        y_pred=self.layer2(y_pred)
        y_pred=torch.relu(y_pred) # 对layer2的输出应用ReLU激活函数
        y_pred=self.layer3(y_pred)
        y_pred=torch.relu(y_pred) # 对layer3的输出应用ReLU激活函数
        return y_pred

net=Net(44,1)
optimizer=torch.optim.Adam(net.parameters(),lr=1e-4) # 定义优化器，使用Adam优化算法，学习率1e-4
loss_func=torch.nn.MSELoss() # 定义损失函数，使用均方误差损失函数

losses=[] # 存储训练损失列表
eval_losses=[] # 存储评估损失列表
for epoch in range(1000):
    train_loss=0.0
    net.train() # 切换到训练模式
    for x,y in train_data:
        y_=net(x) # 前向传播，获取模型输出
        loss=loss_func(y_,y)# 计算损失
        optimizer.zero_grad() # 梯度清零 
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        train_loss=train_loss+loss.item() # 累加训练损失
    losses.append(train_loss/len(train_data)) # 记录训练损失
    eval_loss=0.0
    net.eval() # 切换到评估模式
    with torch.no_grad(): # 禁用梯度计算
        for x,y in test_data:
            y_=net(x) # 前向传播，获取模型输出
            loss=loss_func(y_,y)# 计算损失
            eval_loss=eval_loss+loss.item() # 累加评估损失
    eval_losses.append(eval_loss/len(test_data)) # 记录评估损失
    if epoch%100==0:
        print('Epoch: {}, Train Loss: {:.4f}, Eval Loss: {:.4f}'.format(epoch, train_loss/len(train_data), eval_loss/len(test_data)))

# 6. 评估模型与预测
y_=net(test_features) # 前向传播，获取模型输出
y_pred=y_*std+mean
print("测试集预测值：",y_pred.squeeze().detach().cpu().numpy())
print("模型平均误差",abs(y_pred-(test_labels*std+mean)).mean().cpu().item())
end=time.perf_counter()
print("运行时间：",end-start)



