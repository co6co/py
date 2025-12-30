import  torch
import torch.nn as nn
import torch.utils.data as Data #加载数据集、进行预处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import transformer
from copy import deepcopy as copy #用于深拷贝

# 1. 读取数据
# 使用的模拟数据
region =pd.read_csv("./data/tmp/ssq.csv")
region_d=region[['QH','RQ']] #选取 指定的四列
region['target']=region[['H1','H2','H3','H4','H5','H6','L']]
batch_size=30
dataset=Data.TensorDataset(region_d,region['target'])
# 可通过迭代数据加载器来获取每个批次的数据进行训练
dataLoader=Data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2) # shuffle 是否打乱数据顺序


# 设置神经网络，这里使用一个简单的线性结构
class Net(nn.Module):
    """
    定义一个简单的神经网络模型，包含多个线性层和 ReLU 激活函数。
    使用nn.Module和nn.Sequential 构建神经网络模型 

    这样的结构可以用于构建和使用神经网络模型，对输入数据进行处理和预测，具体的网络结构和参数可以根据需要进行调整和修改
    """
    def __init__(self):
        super(Net,self).__init__()
        # 创建一个序列型神经网络
        self.net=nn.Sequential( 
            nn.Linear(in_features=2,out_features=10),# 输入特征数为1，输出特征数为10的线性层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(10,100), # 输入特征10，输出特征100的线性层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(100,7) ,# 输入特征100，输出特征10的线性层
            
        )
    def forward(self,input:torch.FloatTensor):
        """
        前向传播函数，定义了输入张量 input 如何通过神经网络层进行计算，
        并返回输出张量。
        """
        return self.net(input) # 通过神经网络传播输入数据

net=Net()
model_statue_path='./data/a.lstm.pth'
# 定义损失函数和优化器
# 优化器用于调整模型的参数，以最小化损失函数
# 损失函数用于衡量模型预测值与真实值之间的差异，
# 在训练过程中，通过优化器和损失函数的配合，模型会不断调整参数，以提高预测的准确性
optim=torch.optim.Adam(net.parameters(net),lr=0.001) # Adam 优化器，学习率为0.001
loss_func=nn.MSELoss() # 创建均方误差损失函数


def tran_model():
    best_model=None # 最佳模型
    train_loss=0 # 训练损失
    test_loss=0 #测试损失
    best_loss=100 # 最佳损失
    epoch_cnt=0 #轮数计算
    # 开始训练模型并进行预测 训练100次
    for epoch in range(100):
        total_train_loss=0 # 训练总损失
        total_train_num=0 # 训练样本总数
        total_test_loss=0 # 测试总损失
        total_test_num=0 # 测试样本总数
        for batch_x,batch_y in dataLoader:
            # 前向传播
            pred=net(batch_x) # 将批次数据输入网络进行预测
            
            losser=loss_func(pred,batch_y) # 计算预测结果与真是标签的损失
            # 反向传播和优化
            optim.zero_grad() # 清空优化器梯度
            losser.backward() # 反向传播计算梯度
            optim.step() # 根据梯度更新模型参数
            total_train_loss+=losser.item() 
            total_train_num+=1

        if (epoch+1)%10==0:
            print(f"训练步骤{epoch+1},损失值:{losser.item()}")
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
    if epoch_cnt<early_stop:# 
        torch.save(best_model.state_dict(),model_statue_path)

def test_model(test_dataloader):
    """测试模型的函数
    @param test_dataloader: 测试数据加载器
    @return: 测试值、真是标签和测试损失
    """
    test_loss=0 # 测试损失
    total_test_loss=0 # 测试总损失
    total_test_num=0 # 测试样本总数
    for batch_x,batch_y in test_dataloader:
        # 前向传播
        pred=net(batch_x) # 将批次数据输入网络进行预测
        losser=loss_func(pred,batch_y) # 计算预测结果与真是标签的损失
        total_test_loss+=losser.item() 
        total_test_num+=1
    test_loss=total_test_loss/total_test_num
    return pred,batch_y,test_loss

def plot_test_result(pred,batch_y,test_loss):
    """绘制测试结果的函数
    @param pred: 测试值
    @param batch_y: 真实标签
    @param test_loss: 测试损失
    """ 
    # 绘制预测值与真实值的阵线图
    plt.figure(figsize=(12,7),dpi=160) # 创建一个10x6 英寸的图形窗口,分辨率160dpi
    plt.title("余弦函数拟合") # 设置图形标题
    plt.plot(x,y,label="真实值",marker="X") # 绘制真实值曲线，标签为"真实值"
    plt.plot(x,predict.detach().numpy(),label="预测值",marker="o") # 绘制预测值曲线，标签为"预测值",标记符合为o
    plt.xlabel("x",size=15) # 设置x轴标签 
    plt.ylabel("cos(x)",size=15) # 设置y轴标签
    plt.xticks(size=15) # x 刻度字体 15
    plt.yticks(size=15) # y 刻度字体 15
    plt.legend(fontsize=15) # 显示图例
    plt.show() # 显示图形
    

if __name__ == "__main__":
    # 超参数
    days_num=5
    epoch=20
    fea=5
    batch_size=20
    early_stop=5
    tran_model() 
    # 使用网络对输入数据X进行预测
    predict=net(torch.tensor(X,dtype=torch.float32))
    ## 训练循环和预测代码，主要用于在多轮次训练模型，并每个轮次中计算损失并更新模型参数
    # 这样的训练过程通常用于深度学习模型的训练，通过不断调整模型的参数来最小化损失，以提高模型的性能；
    test_pred,test_y,test_loss=test_model(test_dataLoader)
    plot_test_result(test_pred,test_y,test_loss)
