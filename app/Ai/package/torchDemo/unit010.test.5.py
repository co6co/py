#-*-coding:utf-8-*-
# 练习题
# PyTorch实现Attention模型
import torch
from torch import nn,optim  

class AttentionModel(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(AttentionModel,self).__init__()
        self.hidden_size=hidden_size
        self.encoder=nn.Linear(input_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,1)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,inputs):
        encoded=self.encoder(inputs)
        weights=self.decoder(encoded)
        weights=self.softmax(weights)
        output=torch.matmul(weights.transpose(1,2),encoded).squeeze(1)
        return output

input_size=10
hidden_size=5
input_sequence=torch.randn(3,5,input_size)
attention_model=AttentionModel(input_size,hidden_size)
output=attention_model(input_sequence)
print(output.shape) #输出特征向量的形状
        
