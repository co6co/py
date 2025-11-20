#-*- coding: utf-8 -*-
# 拟合余弦函数曲线
# 使用pytorch 拟合余弦函数曲线，展示预测值和真实值的折线图
from re import X
import torch
import torch.nn as nn # 用于构建神经网络模型
from torch.utils.data import DataLoader #DataLoader 、TensorDataset 用于加载数据
from torch.utils.data import TensorDataset 
import numpy as np
import matplotlib # 用于绘制图形
import matplotlib.pyplot as plt # 用于绘制图形
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
matplotlib.rcParams['axes.unicode_minus'] = False # 用于显示负号，避免出现乱码
 
x=np.linspace(-2*np.pi,2*np.pi,400) # 生成-2π 到 2π 的 400 个点
y=np.cos(x) # 计算余弦函数值    
X=np.expand_dims(x,axis=1) # x扩展一维数组X，即在列方向上增加一个维度
Y=y.reshape(400,-1) #  y 整形为 400行1列的矩阵Y
# X 与 Y 转换为 torch.Tensor 类型，并组成数据集dataset
dataset=TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32))
#创建数据加载器 dataloader，批次大小为10，数据打乱
dataloader=DataLoader(dataset,batch_size=10,shuffle=True)
##
#通过生成数据集和数据加载器，方便对数据进行批量处理和模型训练，
#在深度学习中加载器通常用于将数据分批次加载到模型中进行训练

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
            nn.Linear(in_features=1,out_features=10),# 输入特征数为1，输出特征数为10的线性层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(10,100), # 输入特征10，输出特征100的线性层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(100,10) ,# 输入特征100，输出特征10的线性层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(10,1) # 输入特征10，输出特征1的线性层
        )
    def forward(self,input:torch.FloatTensor):
        """
        前向传播函数，定义了输入张量 input 如何通过神经网络层进行计算，
        并返回输出张量。
        """
        return self.net(input) # 通过神经网络传播输入数据

net=Net()

# 定义损失函数和优化器
# 优化器用于调整模型的参数，以最小化损失函数
# 损失函数用于衡量模型预测值与真实值之间的差异，
# 在训练过程中，通过优化器和损失函数的配合，模型会不断调整参数，以提高预测的准确性
optim=torch.optim.Adam(net.parameters(net),lr=0.001) # Adam 优化器，学习率为0.001
loss=nn.MSELoss() # 创建均方误差损失函数

# 开始训练模型并进行预测 训练100次
for epoch in range(100):
    for batch_x,batch_y in dataloader:
        # 前向传播
        pred=net(batch_x) # 将批次数据输入网络进行预测
        
        l=loss(pred,batch_y) # 计算预测结果与真是标签的损失
        # 反向传播和优化
        optim.zero_grad() # 清空优化器梯度
        l.backward() # 反向传播计算梯度
        optim.step() # 根据梯度更新模型参数
    if (epoch+1)%10==0:
        print(f"训练步骤{epoch+1},损失值:{l.item()}")

# 使用网络对输入数据X进行预测
predict=net(torch.tensor(X,dtype=torch.float32))
## 训练循环和预测代码，主要用于在多轮次训练模型，并每个轮次中计算损失并更新模型参数
# 这样的训练过程通常用于深度学习模型的训练，通过不断调整模型的参数来最小化损失，以提高模型的性能；

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
 