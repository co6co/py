#-*- coding: utf-8 -*-
# 优化器的使用与比较
# 创建一个具有基层的简单神经网络，并使用不同的优化器进行训练，比较他们的性能。
import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def init(self):
        super(SimpleNet,self).init()
        self.fc1=nn.Linear(10,50)
        self.fc2=nn.Linear(50,1)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
# 实例化网络和不同的优化器
net=SimpleNet()
optimizers={
    'SGD': torch.optim.SGD(net.parameters(),lr=0.01),
    'Adam':torch.optim.Adam(net.parameters(),lr=0.001),
    'RMSProp':torch.optim.rmsprop.RMSprop(net.parameters(),lr=0.01),
    'Adagrad':torch.optim.adagrad.Adagrad(net.parameters(),lr=0.01)
}
# 定义损失函数
criterion=nn.MSELoss()
# 生产一些数据进行训练
inputs=torch.randn(100,10)
targets=torch.randn(100,1)

# 训练网络，记录不同优化器的性能
for name,optimizer in optimizers.items():
    net.train()  # 将网络设置为训练模式

    for epoch in range(100):
        inputs.requires_grad=True # 为False 不会有梯度计算
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch+1)%10==0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")


## 通过这个联系，你不仅能学会如何使用PyTorch中的不同优化器，还能直观地看到不同优化器对不同模型训练的影响，
#  也可以进一步分析不同优化器的性能差异，并讨论可能的原因