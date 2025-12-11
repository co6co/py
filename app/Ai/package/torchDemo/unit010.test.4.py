#-*-coding:utf-8-*-
# 练习题
# PyTorch实现Seq2Seq模型
import torch
from torch import nn,optim  

class Seq2Vec(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Seq2Vec,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.rnn=nn.GRU(hidden_size,hidden_size)
        self.fc=nn.Linear(hidden_size,output_size)

        
    def forward(self,input_seq):
        embedded=self.embedding(input_seq)
        output,hidden=self.rnn(embedded)
        output=self.fc(hidden[-1])
        return output

# 创建模型实例
input_size=10000
hidden_size=256
output_size=2
model=Seq2Vec(input_size,hidden_size,output_size)

# 损失优化
critertion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

# 训练
num_epochs=10
for epoch in range(num_epochs):
    for input_seq,target in train_data: #假设train_data是一个可迭代对象，每个元素是一个(input_seq,target)对
        optimizer.zero_grad()
        output=model(input_seq)
        loss=critertion(output,target)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    