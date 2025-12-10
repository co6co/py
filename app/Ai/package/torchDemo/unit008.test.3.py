#-*- coding: utf-8 -*-
# 实现交叉验证
# 十折交叉验证
# 将数据分成10组，进行10组训练
# 每组用于测试的数据为：数据总条数/组数，每次测试的数据都是随机抽取的


import torch
import numpy as np
from torch._dynamo.convert_frame import output_codes
from torch.utils.data import DataLoader,Dataset,TensorDataset
import time
import torch.nn.functional as F
import torch.nn as nn

start=time.perf_counter()

# 1. 构造训练集
x=torch.rand(100,28,28) # 生产形状为（100,28,28）的随机张量x
y=torch.randn(100,28,28)
x=torch.cat((x,y),dim=0) # 在维度0上连接x、y
label=[1]*100+[0]*100 # 生成长度为200的标签
label=torch.tensor(label,dtype=torch.long) # 将列表label转换为张量

# 2. 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(28*28,120) # 全连接层
        self.fc2=nn.Linear(120,84) # 全连接层
        self.fc3=nn.Linear(84,2) # 全连接层
    def num_flat_features(self,x):
        """
        计算输入张量x的展平特征数
        """
        size=x.size()[1:] #获取输入x的除第一个维度外的尺寸
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    def forward(self,x):
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) 
        return x

# 3.训练集数据处理
class TrainDataSet(Dataset):
    def __init__(self,train_features,train_labels):
        self.x=train_features
        self.y=train_labels
        self.len=len(train_labels)
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        return self.x[index],self.y[index]

# 4.设置损失函数
loss_func=nn.CrossEntropyLoss()

# 5. 设置K折划分
def get_k_fold_data(k:int,i:int,X:torch.Tensor,y:torch.Tensor):
    """
    获取k折交叉验证数据
    @param k: 折数
    @param i: 当前折数索引
    @param X: 输入数据
    @param y: 标签
    返回：
    X_train:torch.Tensor 训练集数据
    y_train:torch.Tensor 训练集标签
    X_valid:torch.Tensor 验证集数据
    y_valid:torch.Tensor 验证集标签
    """
    assert k>1
    fold_size=X.shape[0]//k # 每折数据量
    X_train,y_train=None,None 
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size) # 第j折的索引范围
        X_part,y_part=X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid=X_part,y_part 
        elif X_train is None:
            X_train,y_train=X_part,y_part 

        else:
            X_train=torch.cat((X_train,X_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return X_train,y_train,X_valid,y_valid

def k_fold(k:int,X_train:torch.Tensor,y_train:torch.Tensor,num_epochs:int=3,learning_rate=0.001,weight_decay=0.1,batch_size=5):
    """
    k折交叉验证
    @param k 折数
    @param X_train 训练集数据
    @param y_train 训练集标签
    @param num_epochs 训练轮数
    @param learning_rate 学习率
    @param weight_decay 权重衰减
    @param batch_size 批量大小
    返回：
    train_loss_sum:float 训练集损失总和
    valid_loss_sum:float 验证集损失总和
    train_acc_sum:float 训练集准确度总和
    valid_acc_sum:float 验证集准确度总和
    """
    train_loss_sum,valid_loss_sum,train_acc_sum,valid_acc_sum=0.0,0.0,0.0,0.0
    for i in range(k):
        data=get_k_fold_data(k,i,X_train,y_train)
        net=Net()
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        print("*"*10,f"第{i+1}折",'*'*10,train_ls,valid_ls)
        print(f"训练集损失：{train_ls[-1][0]:.6f}训练集准确度：{valid_ls[-1][1]:.4f},测试集损失：{valid_ls[-1][0]:.6f}测试集准确度：{valid_ls[-1][1]:.4f} ")
        train_loss_sum+=train_ls[-1][0]
        valid_loss_sum+=valid_ls[-1][0]
        train_acc_sum+=train_ls[-1][-1]
        valid_acc_sum+=valid_ls[-1][-1]
    print("*"*5,'最终k折交叉验证结果',"#"*5)
    print(f"训练集损失：{train_loss_sum/k:.4f}训练集准确度：{train_acc_sum/k:.4f},测试集损失：{valid_loss_sum/k:.6f}测试集准确度：{valid_acc_sum/k:.4f} ")

# 6.设置训练函数
def train(net,train_features,train_labels,test_features,test_labels,num_epochs=3,learning_rate=0.001,weight_decay=0.1,batch_size=5):
    """
    训练网络
    @param net 网络模型
    @param train_features 训练集数据
    @param train_labels 训练集标签
    @param test_features 验证集数据
    @param test_labels 验证集标签
    @param num_epochs 训练轮数
    @param learning_rate 学习率
    @param weight_decay 权重衰减
    @param batch_size 批量大小
    返回：
    train_ls:list 训练集损失列表
    test_ls:list 验证集损失列表 
    """
    train_ls,test_ls=[],[]
    dataset=TrainDataSet(train_features,train_labels)
    train_iter=DataLoader(dataset,batch_size,shuffle=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    for epoch in range(num_epochs): 
        for X,y in train_iter:
            y_hat=net(X)
            loss=loss_func(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_ls.append(log_rmse(0,net,test_features,test_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(1,net,test_features,test_labels))
    return train_ls,test_ls

def log_rmse(flag,net:Net,x:torch.Tensor,y:torch.Tensor):
    """
    计算模型在特征集上的均方根误差（RMSE）
    @param flag 标志位，0表示训练集，1表示测试集
    @param net 网络模型
    @param x 特征集
    @param y 标签集
    返回：
        (损失值,准确度)  
    """
    if flag==1:
        net.eval() # 如果是测试集，将模型设置为评估模型

    output=net(x) #进行向前传播得到输出
    result=torch.max(output,1)[1].view(y.size(  )) # 取输出中最大值的索引作为预测结果
    corrects=(result.data==y.data).sum().item() # 计算正确预测的数量
    accuracy=corrects*100.0/len(y) #计算准确度
    loss=loss_func(output,y)
    net.train() # 将模型设置回训练模式
    return (loss.data.item(),accuracy)
    
# 7 调用交叉验证函数
k_fold(10,x,label) # 执行10折交叉验证
        
