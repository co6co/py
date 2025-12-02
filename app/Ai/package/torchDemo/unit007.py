#-*- coding: utf-8 -*-
# 神经网络模型
# 使用长短期记忆网络模型对上海证券交易所工商银行的股票成交量做一个趋势预测，
# #这样可以更好的掌握股票的买卖点，从而提高自己的收益率。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tushare as ts # 用于获取股票数据 pip install tushare
import matplotlib.pyplot as plt # 用于绘制图像
from copy import deepcopy as copy #用于深拷贝
from torch.utils.data import Dataset, DataLoader,TensorDataset #加载数据集
import pandas as pd # 用于数据处理
from tqdm import tqdm # 用于进度条显示 pip install tqdm

# 1. 获取数据 通过 tushare库获取股票数据，

# 使用开盘价、收盘价、最高价、最低价、成交量这5个特征。
# 使用每天的收盘价作为学习目标，每个样本都包含连续几天的数据作为一个序列样本
# 最后将这些数据划分为训练集和测试供后续使用
class GetData:
    def __init__(self,stock_id,save_path):
        """初始化方法
        @param stock_id: 股票id
        @param save_path: 保存路径
        """
        self.stock_id=stock_id
        self.save_path=save_path
        self.data=None
    def getData(self):
        """获取数据
        @return: 处理后的数据
        """
        if self.save_path:
            try:
                self.data=pd.read_csv(self.save_path)
            except:
                self.data=ts.get_k_data(self.stock_id).iloc[::-1]
        # 选择特定列作为数据
        self.data=self.data[['open','close','high','low','volume']]
        self.close_min=self.data['volume'].min()
        self.close_max=self.data['volume'].max()
        # 归一化
        self.data=self.data.apply(lambda x:(x-x.min())/(x.max()-x.min()))
        # 保存数据
        self.data.to_csv(self.save_path,index=False)
        return self.data
    def process_data(self,n):
        """处理数据
        将数据分为特征和标签，并划分为训练集和测试集
        @param n: 滑动窗口大小
        @return: 训练集的特征，测试集特征、训练集标签和测试集标签
        """
        if self.data is None:
            self.getData()
        feature=[
            self.data.iloc[i:i+n].values.tolist()
            for i in range(len(self.data)-n+2)
            if i+n<len(self.data)
        ]
        label=[
            self.data.close.values[i+n]
            for i in range(len(self.data)-n+2)
            if i+n<len(self.data)
        ] 
        # 划分训练集和测试集 
        train_x=feature[:500]
        train_y=label[:500]
        test_x=feature[500:]
        test_y=label[500:]
        return train_x,test_x,train_y,test_y

# 2. 搭建LSTM模型，使用单层单向LSTM网络，加一个全连接层输出
model_statue_path='./data/007.lstm.pth'
##　定义神经网络模型
class Model(nn.Module):
    def __init__(self,n):
        super(Model,self).__init__()
        # 创建一个LSTM层，输入大小为n，隐藏大小为256，批次优先为True
        self.lstm_layer=nn.LSTM(input_size=n,hidden_size=256,batch_first=True)
        # 创建一个线性层，输入特征数为256，输出特征数为1，有偏差
        self.linear_layer=nn.Linear(in_features=256,out_features=1,bias=True)
    def forward(self,x): # 向前传播方法，接受一个输入x
        out1,(h_n,h_c)=self.lstm_layer(x) # 通过LSTM层处理x，得到输出out1和隐藏状态h_n、h_c
        a,b,c=h_n.shape # 获取h_n的形状 a: 批次大小, b: 隐藏层大小, c: 隐藏层数量
        # 将h_n 重塑（a*b,c）的形状后，通过线性层处理得到输出 out2
        out2=self.linear_layer(h_n.reshape(a*b,c))
        return out2 # 返回最终输出out2
# 3. 训练模型
##　计算损失ｌｏｓｓ，损失backward以及优化器step
def train_model(epoch,train_dataloader,test_dataLoader):
    """训练模型的函数
    @param epoch: 训练轮数
    @param train_dataloader: 训练数据加载器
    @param test_dataLoader: 测试数据加载器
    """
    best_model=None # 最佳模型
    train_loss=0 # 训练损失
    test_loss=0 #测试损失
    best_loss=100 # 最佳损失
    epoch_cnt=0 #轮数计算
    for _ in range(epoch):
        total_train_loss=0 # 训练总损失
        total_train_num=0 # 训练样本总数
        total_test_loss=0 # 测试总损失
        total_test_num=0 # 测试样本总数
        for x,y in tqdm(train_dataloader,desc='Epoch:{}|Train Loss:{}|Test Loss:{}'.format(_,train_loss,test_loss)):
            # x 的数量
            x_num=len(x)
            p=model(x) # 模型预测
            #print(len(p[0]))
            loss=loss_func(p,y)
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_train_loss+=loss.item() 
            total_train_num+=x_num
        # 计算平均训练损失
        train_loss=total_train_loss/total_train_num
        for x,y in test_dataLoader:
            x_num=len(x)
            p=model(x) # 模型预测
            loss=loss_func(p,y)
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_test_loss+=loss.item()
            total_test_num+=x_num
        # 计算平均测试损失
        test_loss=total_test_loss/total_test_num
        if test_loss<best_loss:
            best_loss=test_loss
            best_model=copy(model)
        else:
            epoch_cnt+=1
        if epoch_cnt>=early_stop:
            # 保存最佳模型的状态字典
            torch.save(best_model.state_dict(),model_statue_path)
            break



# 4. 测试模型
##  用测试模型进行预测
def test_model(test_dataloader):
    """测试模型的函数
    @param test_dataloader: 测试数据加载器
    @return: 测试值、真是标签和测试损失
    """
    # 预测值列表
    pred=[]
    #真实标签列表
    label=[]
    # 创建模型对象
    model_=Model(5)
    model_.load_state_dict(torch.load(model_statue_path))
    model_.eval() # 将模型设置为评估模式
    total_test_loss=0
    total_test_num=0
    for x,y in test_dataloader:
        x_num=len(x)
        p=model_(x) # 模型预测
        loss=loss_func(p,y)
        total_test_loss+=loss.item()
        total_test_num+=x_num
        pred.append(p.data.squeeze(1).tolist())
        label.append(y.tolist())
    test_loss=total_test_loss/total_test_num
    return pred,label,test_loss

def plot_img(data,pred):
    """绘制图像
    @param data: 数据
    @param pred: 预测值
    """
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.figure(figsize=(12,7))
    plt.plot(range(len(pred)),pred,color='green') # 绘制预测值,颜色绿
    plt.plot(range(len(data)),pred,color='blue') # 绘制数据曲线,颜色蓝
    for i in range(0,len(pred)-3,5):
        price=[data[i]+pred[j]-pred[i] for j in range(i,i+3)]
        plt.plot(range(i,i+3),price,color='red') 
    plt.xticks(fontproperties='Times New Roman',size=15)
    plt.yticks(fontproperties='Times New Roman',size=15)
    plt.xlabel('日期', size=15)
    plt.ylabel('成交量', size=15) 
    plt.show()

if __name__ == "__main__":
    # 超参数
    days_num=5
    epoch=20
    fea=5
    batch_size=20
    early_stop=5

    #初始化模型
    model=Model(fea)
    # 处理数据
    GD=GetData(stock_id='601398',save_path='./data/007.601398.csv')
    x_train,x_test, y_train,y_test=GD.process_data(days_num)
    x_train=torch.tensor(x_train).float()
    x_test=torch.tensor(x_test).float()
    y_train=torch.tensor(y_train).float()
    y_test=torch.tensor(y_test).float()
    # 创建训练数据加载器
    train_data=TensorDataset(x_train,y_train)
    train_dataloader=DataLoader(train_data,batch_size=batch_size )

    test_data=TensorDataset(x_test,y_test)
    test_dataloader=DataLoader(test_data,batch_size=1,shuffle=False)
    
    # 定义损失函数和优化器
    loss_func=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    # 训练模型
    train_model(epoch,train_dataloader,test_dataloader)
    p,y,test_loss=test_model(test_dataloader)
    # 对预测值进行处理
    pred=[ele*(GD.close_max-GD.close_min)+GD.close_min for ele in p]
    data=[ele*(GD.close_max-GD.close_min)+GD.close_min for ele in y] 
    # 绘制图像
    plot_img(data,pred)

     

