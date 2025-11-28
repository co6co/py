#-*- coding: utf-8 -*-
#优化器
# PYTorch 中有SGD、Momentum、AdaGrad、 RMSprop、Adam等优化器
# 优化器繁多，应选择合适的优化器
# 下面示例给出SGD优化器的使用 
import torch
import torch.nn as nn # 神经网络模块，用于定义神经网络模型结构
import torch.utils.data as Data #加载数据集、进行预处理
import matplotlib #绘制图标和可视化结果
import matplotlib.pyplot as plt #绘制模块，用于绘制图标
matplotlib.rcParams['font.sans-serif']=['SimHei'] #设置字体，支持中文

# 1. 准备建模数据
x=torch.unsqueeze( torch.linspace(-1,1,500),dim=1)
y=x.pow(3) # x^3

#2. 设置超参数
LR=0.01 # 学习率，决定模型在每次更新参数时的调整幅度
batch_size=15 # 批次大小，每次训练时同时处理样本的数量
epochs=5 # 训练轮数，即对整个数据集进行训练的次数
torch.manual_seed(10) # 设置随机种子，确保结果可重复

# 3. 设置加载器
dataset=Data.TensorDataset(x,y)
# 可通过迭代数据加载器来获取每个批次的数据进行训练
loader=Data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2) # shuffle 是否打乱数据顺序

# 4. 搭建神经网络框架
class Net(nn.Module):
    """
    由一个隐藏层和一个输出层组成的神经网络模型Net
    """
    def __init__(self,n_input,n_hidden,n_output):
        """通过 nn.Linear 定义了隐层和输出层的线性变换"""
        super(Net,self).__init__()
        self.hidden_layer=nn.Linear(n_input,n_hidden) # 隐藏层
        self.output_layer=nn.Linear(n_hidden,n_output) # 输出层
        pass
    def forward(self,x):
        """定义了向前传播的计算过程，
        输入数据通过隐藏层进行线性变换后，应用激活函数，
        再经过输出层的线性变换得到最终结果

        这样的模型结构可用于各种任务：如分类、回归等，
        可以将输入数据input传入Net对象的forward方法得到模型的输出。
        可根据需要进行训练、评估和预测等操作.

        """
        x=torch.relu(self.hidden_layer(x)) # 通过隐藏曾进行线性变化，并应用ReLU激活函数
        output=self.output_layer(x)
        return output 

# 5. 训练模型，绘制折线图
def train():
    """定义训练函数
    训练多个不同优化器的神经网络模型
    绘制损失曲线
    """
    # 创建不同的神经网络模型
    net_SGD=Net(1,10,1)
    net_momentum=Net(1,10,1)
    net_AdaGrad=Net(1,10,1)
    net_RMSprop=Net(1,10,1)
    net_Adam=Net(1,10,1)

    nets=[net_SGD,net_momentum,net_AdaGrad,net_RMSprop,net_Adam]
    # 定义不同的优化器
    optimizers=[
        # 随机梯度下降优化器
        torch.optim.SGD(net_SGD.parameters(),lr=LR),
        # 带动量的随机梯度下降优化器
        torch.optim.SGD(net_momentum.parameters(),lr=LR,momentum=0.6),
        # AdaGrad优化器
        torch.optim.Adagrad(net_AdaGrad.parameters(),lr=LR,lr_decay=0.9),
        # RMSProp 优化器
        torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9),
        #Adam 优化器
        torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    ]
    # 定义损失函数
    loss_func=torch.nn.MSELoss() # 均方误差损失函数
    losses=[[] for _ in range(len(nets))] # 用于存储每个模型的损失值
    # 训练模型
    for epoch in range(epochs):
        for step,(batch_x,batch_y) in enumerate(loader):
            # 按模型、优化器和损失函数顺序循环
            for net,optimizer,loss_list in zip(nets,optimizers,losses):
                prediction_y=net(batch_x) # 向前传播得到预测值
                loss=loss_func(prediction_y,batch_y) # 计算损失
                optimizer.zero_grad() # 清空梯度
                loss.backward() # 反向传播
                optimizer.step() # 更新参数
                #lesses[nets.index(net)].append(loss.item()) # 记录损失值
                loss_list.append(loss.data.item()) # 记录损失值
 

    # 创建图像
    plt.figure(figsize=(12,7))
    # 定义标签
    labels=['SGD','momentum','AdaGrad','RMSprop','Adam']
    for i,loss in enumerate(losses): # 按模型顺序循环
        plt.plot(loss,label=labels[i]) # 绘制每个模型的损失曲线
        plt.title('不同优化器的损失曲线',fontsize=12)
        # 增加图例
        plt.legend(loc='upper right',fontsize=12)
        # 设置刻度标签大小
        plt.tick_params(labelsize=12)
        # 设置x轴标签
        plt.xlabel('训练步骤',fontsize=15)
        # 设置y轴标签
        plt.ylabel('模型损失',fontsize=15)
        plt.ylim(0,0.3)# 设置Y轴范围
        
    plt.show()

if __name__=='__main__':
    train()


