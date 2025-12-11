#-*- coding: utf-8 -*-
#Attention 模型文本自动分类

from multiprocessing.util import abstract_sockets_supported
import random,math,sys,time,os
from collections import Counter
from PIL.ImageFont import MAX_STRING_LENGTH
from regex import B
from sympy import evaluate
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch import nn,optim 
import torch.nn.functional as F
import torchtext

from unit009 import criterion


# 加载数据
batch_size=128
leaning_rate=le-3
embedding_dim=100
torch.manual_seed(99) # 随机种子
text=torchtext.legacy.data.Field(tokenize=lambda x:x.split(),tokenize='spacy',lower=True)
label=torchtext.legacy.data.LabelField(dtype=torch.float)

def get_dataset(corpur_apth,text_field,label_field):
    """
    加载数据集
    @param corpur_apth: 数据集路径
    @param text_field:  torchtext Field 对象 文本字段
    @param label_field: 标签字段
    @return: 数据集
    examples:list 
    fields:tuple
    """
    fields=[('label',label_field),('text',text_field)]
    examples=[]
    with open(corpur_apth,'r',encoding='utf-8') as f:
        li=[]
        while True:
            line=f.readline().replace('\n','')
            if not line:
                
             
                if not li:
                    break
                label=li[0][10]
                text=li[1][6:-7] 
                examples.append(torchtext.legacy.data.Example.fromlist([label,text],fields))
            else:
                li.append(line)
    return examples,fields 

# 获取训练集/测试集 示例和字段 
train_examples,train_fields=get_dataset('./data/tmp/10.2.trains.txt',text,label)
dev_examples,dev_fields=get_dataset('./data/tmp/10.2.devs.txt',text,label)
test_examples,test_fields=get_dataset('./data/tmp/10.2.tests.txt',text,label)

train_data=torchtext.legacy.data.Dataset(train_examples,train_fields)
dev_data=torchtext.lagacy.data.Dataset(dev_examples,dev_fields)
test_data=torchtext.legacy.data.Dataset(test_examples,test_fields)
print(f'训练集样本数: {len(train_data)}')
print(f'验证集样本数: {len(dev_data)}')
print(f'测试集样本数: {len(test_data)}')

# 创建词向量
text.build_vocat(train_data,max_size=5000,vectors='glove.6B.100d')
label.build_vocat(train_data) #在训练集上构建标签词汇
print("训练集上词向量的数量",len(test.vocab))
# 创建迭代器
#根据数据集创建迭代器
train_iterator,dev_iterator,test_iterator=torchtext.legacy.data.BucketIterator.splits((train_data,dev_data,test_data),batch_size=batch_size,sort=False)

# 网络模型
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size,embedding_dim,hidden_dim,n_layers) -> None:
        """
        @param vocab_size int :词汇表大小
        @embedding_dim int :嵌入维度
        @hidden_dim int 隐层层维度
        @n_layers int LSTM层数量
        """
        super(BiLSTM_Attention,self).__init__()
        self.hiddent_dim=hidden_dim
        self.n_layers=n_layers
        self.embedding=nn.Embedding(vocab_size,embedding_dim) # 嵌入层
        self.rnn=nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,bidirectional=True,dropout=0.5)
        self.fc=nn.Linear(hidden_dim*2,1)
        self.dropout=nn.Dropout(0.5)
        self.w_omega=nn.Parameter(torch.Tensor(hidden_dim*2,hidden_dim*2)) # 注意力权重矩阵
        self.u_omega=nn.Parameter(torch.Tensor(hidden_dim*2,1)) 
        nn.init.uniform_(self.w_omega,-0.1,0.1)# 初始化注意力权重矩阵
        nn.init.uniform_(self.u_omega,-0.1,0.1) # 初始化注意力偏差向量
    
    def attention_net(self,x:torch.Tensor):
        """
        计算注意力机制的输出
        x 输入张量
        返回：
        context torch.Tensor 注意力机制的输出
        """
        u=torch.tanh(torch.matmul(x,self.w_omega)) # 计算tanh激活后的注意力得分
        att=torch.matmul(u,self.u_omega) #计算注意力得分
        att_score=F.softmax(att,dim=1) # 在最后一维上进行Softmax操作
        scored_x=x*att_score #乘以注意力得分
        context=torch.sum(scored_x,dim=1) # 在第一维上求和
        return context
    def forward(self,x:torch.Tensor):
        """
        前向传播
        x 输入张量
        返回：
        out torch.Tensor 输出张量
        """
        emb=self.dropout(self.embedding(x)) # 嵌入层
        out,(h_n,c_n)=self.rnn(emb) # LSTM层
        out=out.permute(1,0,2) # 交换维度
        out=self.attention_net(out)#计算注意力机制的输出
        out=self.fc(out) #全连接层的输出
        return out

# 应用
rnn=BiLSTM_Attention(len(text.vocab),embedding_dim,hidden_dim=64,n_layers=2)
#预测训练的嵌入向量
pretrained_embedding=text.vocab.vectors
print("预训练嵌入向量：",pretrained_embedding.shape)
rnn.embedding.weight.data.copy_(pretrained_embedding)# 与训练的嵌入向量复制到rnn的嵌入层权重中
print('嵌入层已初始化')
# 优化器
optimizer=optim.Adam(rnn.parameters(),lr=leaning_rate)
#损失函数
criterion=nn.BCEWithLogitsLoss()


# 训练网络
## 计算准确率
def binary_acc(preds,y):
    """
    计算二分类准确率
    @param preds torch.Tensor 预测值
    @param y torch.Tensor 真实值
    @return float 准确率
    """
    # 对预测值进行四舍五入
    rounded_preds=torch.round(torch.sigmoid(preds))
    # 计算正确预测的数量
    correct=torch.eq(rounded_preds,y).float()
    # 计算准确率
    acc=correct.sum()/len(correct)
    return acc 

# 训练模型
def train(model,iterator,optimizer,criterion):
    """
    训练模型
    @param model nn.Module 模型
    @param iterator torchtext.legacy.data.Iterator 迭代器
    @param optimizer optim.Optimizer 优化器
    @param criterion nn.Module 损失函数
    @return float 平局损失
    @return float 平局准确率
    """
    avg_loss=[]
    avg_acc=[]
    model.train() # 切换到训练模式
    for batch_idx,batch in enumerate(iterator):
        pred=model(batch.text).squeeze() 
        loss=criterion(pred,batch.label) 
        acc=binary_acc(pred,batch.label).item()
        avg_loss.append(loss.item())
        avg_acc.append(acc)
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
    avg_loss=np.array(avg_loss).mean()
    avg_acc=np.array(avg_loss).mean()
    return avg_loss,avg_acc

# 评估模型
def evaluate(rnn,iterator,criteon):
    """
    评估模型
    @param rnn 模型
    iterator 
    criteon 损失函数

    return :
    avg_loss float 平均损失
    avg_acc float 平均准确率 
    """
    avg_loss=[]
    avg_acc=[]
    rnn.eval()
    with torch.no_grad():
        for batcn in iterator:
            pred=rnn(batch.text).squeeze()
            loss=criteon(pred,batch.label)
            acc=binary_acc(pred,batch.label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)
    avg_loss=np.array(avg_loss).mean()
    avg_acc=np.array(avg_acc).mean()
    return avg_loss,avg_acc


# 训练模型
best_valid_acc=float('-inf')
for epoch in range(n_epochs):
    start_time=time.time()
    train_loss,train_acc=train(rnn,train_iterator,optimizer,criterion)
    valid_loss,valid_acc=evaluate(rnn,valid_iterator,criterion)
    end_time=time.time()
    # 计算epoch 花费的时间
    epoch_mins,epoch_secs=divmod(end_time-start_time,60)
    if valid_acc>best_valid_acc:
        best_valid_acc=valid_acc
        torch.save(rnn.state_dict(),'./data/tmp/10.2.best_model.pt')
    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

# 应用模型
rnn.load_state_dict(torch.load('./data/tmp/10.2.best_model.pt'))
test_loss,test_acc=evaluate(rnn,test_iterator,criterion) # 评估模型在测试集上的表现
print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

