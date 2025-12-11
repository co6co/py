#-*- coding: utf-8 -*-
# 机器翻译
# 使用Seq2Seq模型
# 数据集分为训练集和测试集两个文件
import random,math,sys,time,os
from collections import Counter
from PIL.ImageFont import MAX_STRING_LENGTH
from sympy import evaluate
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch import nn,optim 
import torch.nn.functional as F
import nltk

def load_data(file_path):
    """
    加载数据集文件
    :param file_path: 数据集文件路径
    :return: en:list,cn:list
    """
    en,cn=[],[]
    num_examples=0
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip().split('\t')
            en.append(['BOS']+nltk.word_tokenize(line[0].lower())+['EOS'])
            cn.append(['BOS']+nltk.word_tokenize(line[0].lower())+['EOS']) 
            num_examples+=1
    return en,cn 

train_file='./data/tmp/010.1/train.txt'
test_file='./data/tmp/010.1/test.txt'
en_train,cn_train=load_data(train_file)
en_test,cn_test=load_data(test_file)

UNK_IDX=0
PAD_IDX=1
def build_dict(sentences,max_words=50000):
    """
    构建单词到索引的映射
    :param sentences: 句子列表
    :param max_words: 最大单词数
    :return: 
        word_dict:dict 单词到索引的映射字典
        total_words:int 总单词数
    """
    word_count=Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word]+=1
    ls=word_count.most_common(max_words)
    total_words=len(ls)+2
    word_dict={word:idx+2 for idx,word in enumerate(ls)}
    word_dict['UNK']=UNK_IDX
    word_dict['PAD']=PAD_IDX
    return word_dict,total_words

# 调用函数并创建逆字典
en_dict,en_total_words=build_dict(en_train)
cn_dict,cn_total_words=build_dict(cn_train)
en_inv_dict={idx:word for word,idx in en_dict.items()}
cn_inv_dict={idx:word for word,idx in cn_dict.items()}

def encode(en_sentences,cn_sentences,en_dict,cn_dict,sort_by_len=True):
    """
    将英文句子和中文句子编码为索引序列
    @param en_sentences: 英文句子列表
    @param cn_sentences: 中文句子列表
    @param en_dict: 英文单词到索引的映射字典
    @param cn_dict: 中文单词到索引的映射字典
    @param sort_by_len: 是否根据句子长度排序
    @return: 
        en_indices:list 英文句子索引序列列表
        cn_indices:list 中文句子索引序列列表
    """
    length=len(en_sentences)
    out_en_sentences=[[en_dict.get(word,UNK_IDX) for word in sentence] for sentence in en_sentences]
    out_out_sentences=[[cn_dict.get(word,UNK_IDX) for word in sentence] for sentence in cn_sentences]
    
    if sort_by_len:
        sorted_indices=sorted(range(length),key=lambda i:len(out_en_sentences[i]))
        out_en_sentences=[out_en_sentences[i] for i in sorted_indices]
        out_out_sentences=[out_out_sentences[i] for i in out_out_sentences]
    return out_en_sentences,out_out_sentences

en_train,cn_train=encode(en_train,cn_train,en_dict,cn_dict)
en_test,cn_test=encode(en_test,cn_test,en_dict,cn_dict)


# 全部句子分批
def get_minibatches(n,minibatch_size,shuffle=True):
    """
    将n个样本分为多个小批量
    @param n: 样本总数
    @param minibatch_size: 小批量大小
    @param shuffle: 是否打乱样本顺序
    @return: 
        idex_list:list 包含每个最小批次起始索引的列表
        minibatches:list 每个最小批次索引范围的列表
    """
    idx_list=np.arange(0,n,minibatch_size) 
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches=[]
    for idx in idx_list:
        minibatches.append(np.arange(idx,min(idx+minibatch_size,n)))
    return minibatches
# 准备数据
def prepare_data(seqs):
    """
    将序列数据转换为适合模型输入的格式
    参数：
    @param seqs list 序列数据列表
    返回：
    x:ndarray 输入数据矩阵，每个序列按长度填充到最大长度
    x_lengths:ndarray
    """
    lengths=[len(seq) for seq in seqs] # 计算每个序列的长度
    n_samples=len(seqs) #序列总数
    max_len=np.max(lengths)
    x=np.zeros((n_samples,max_len)).astype('int32')
    x_lengths=np.array(lengths).astype("int32")

    for ide,seq in enumerate(seqs):
        x[ide,:lengths[ide]]=seq
        
    return x,x_lengths

def gen_examples(en_sentences,cn_sentences,batch_size):
    """
    生成批量训练示例
    @param en_sentences: 英文句子列表
    @param cn_sentences: 中文句子列表
    @param batch_size: 批量大小
    @return: 
        all_ex:list 包含所有批次数据的列表
    """
    minibatches=get_minibatches(len(en_sentences),batch_size)
    all_ex=[]
    for minibatch in minibatches:
        en_batch=[en_sentences[i] for i in minibatch]
        cn_batch=[cn_sentences[i] for i in minibatch]
        en_batch,en_batch_lengths=prepare_data(en_batch)
        cn_batch,cn_batch_lengths=prepare_data(cn_batch)
        all_ex.append((en_batch,en_batch_lengths,cn_batch,cn_batch_lengths)) 
    return all_ex


batch_size=64
train_data=gen_examples(en_train,cn_train,batch_size)
test_data=gen_examples(en_test,cn_test,batch_size)


# 网络模型
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion,self).__init__()
    def forward(self,input,target,mask):
        # 输入转换为连续的行政，并调整其形状
        input=input.contiguous().view(-1,input.size(2))
        targe=target.contiguous().view(-1,1)
        mask=mask.contiguous().view(-1,1)
        output=-input.gather(1,target)*mask
        output=torch.sum(output)/torch.sum(mask)
        return output

class PlainEncoder(nn.Module):
    def __init__(self, vocab_size,hidden_size,dropout=0.2) -> None:
        super(PlainEncoder,self).__init__( )
        self.embed=nn.Embedding(vocab_size,hidden_size)
        self.rnn=nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.dropout=nn.Dropout(dropout) #创建Dropout层，用于防止过拟合
    def forward(self,x,lengths):
        # 按长度降序对输入进行排序
        sorted_len,sorted_idx=lengths.sort(0,descending=True)
        x_sorted=x[sorted_idx.long()] # 根据排序后的索引重新排序输入
        embedded=self.dropout(self.embed(x_sorted))
        # 将嵌入向量打包成填充的序列
        packed_embeded=nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().cpu().data.numpy(),batch_first=True)
        # 运行循环神经网络
        outputs,hidden= self.rnn(packed_embeded)
        # 解包填充的序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # 恢复原始顺序
        _, original_idx = sorted_idx.sort(0, descending=False)
        outputs = outputs[original_idx.long()].contiguous()
        hidden = hidden[:, original_idx.long()].contiguous()
        return outputs,hidden[[-1]]

# 定义一个名为PlainDecoder 的神经网络模块
class PlainDecoder(nn.Module):
    def __init__(self, vocab_size,hidden_size,dropout=0.2) -> None:
        super(PlainDecoder,self).__init__( )
        self.embed=nn.Embedding(vocab_size,hidden_size)
        self.rnn=nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,hidden_size)
        self.dropout=nn.Dropout(dropout) #创建Dropout层，用于防止过拟合
        self.linear=nn.Linear(hidden_size,vocab_size)
    def forward(self,y,y_lengths,hid):
        sorted_len,sorted_idx=y_lengths.sort(0,descending=True)
        y_sorted=y[sorted_idx.long()]
        hid=hid[:,sorted_idx.long()]
        y_sorted=self.dropout(self.embed(y_sorted))
        packed_seq=nn.utils.rnn.pack_padded_sequence(y_sorted,sorted_len.long().cpu().data.numpy(),batch_first=True)
        out,hid=self.rnn(packed_seq,hid)
        # 解包填充的序列
        unpacked,_=nn.utils.rnn.pack_padded_sequence(out,batch_first=True)
        # 恢复原始顺序
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq=unpacked[original_idx.long()].contiguous()
        hid=hid[:,original_idx.long()].contiguous()
        output=F.log_softmax(self.fc(output_seq),-1)  
        return output,hid

# 构建Seq2Seq模型把encoder、attention和decoder串到一起
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PlainSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder) -> None:
        super(PlainSeq2Seq,self).__init__( )
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,x,x_lengths,y,y_lengths):
        enc_outputs,enc_hid=self.encoder(x,x_lengths)
        dec_outputs,dec_hid=self.decoder(y,y_lengths,enc_hid)
        return dec_outputs,None
    def translate(self,x,x_lengths,y,max_kength=10):
        
        encodeer_cut,hid=self.encoder(x,x_lengths) #运行解码器
        preds=[ ] # 生成的翻译结果
        batch_size=x.shape[0]
        attns=[] # 存储注意力权重
        for i in range(max_kength):
            output,hid=self.decoder(y,y_lengths=torch.ones(batch_size).long().to(device),hid=hid)
            y=output.max(2)[1].view(batch_size,1)
            
            preds.append(y) 
        return torch.cat(preds,1),None #将预测结果沿着维度1拼接成一张量
# 定义模型、损失、优化器
dropout=0.2
hidden_size=100
encoder=PlainEncoder(vocab_size=en_total_words,hidden_size=hidden_size,dropout=dropout)
decoder=PlainDecoder(vocab_size=cn_total_words,hidden_size=hidden_size,dropout=dropout)

model=PlainSeq2Seq(encoder,decoder).to(device)
loss_fn=LanguageModelCriterion()
optimizer=optim.Adam(model.parameters())

# 训练
def train(model,data,num_epoochs=20): 
    for epoch in range(num_epochs):
        model.train()
        total_num_words=total_loss=0. 
        for it,(mb_x,mb_x_len,mb_y,mb_y_len ) in enumerate(data):
            mb_x=torch.from_numpy(mb_x).to(device).long()
            mb_x_len=torch.from_numpy(mb_x_len).to(device).long() 

            mb_input=torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_output=torch.from_numpy(mb_y[:,1:]).to(device).long()
            
            mb_y_len=torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0]=1
            mb_pred,attn=model(mb_x,mb_x_len,mb_input,mb_y_len)
            mb_out_mask=torch.arange(mb_y_len.max().item(),device=device)[None,:]<mb_y_len[:,None]
            mb_out_mask=mb_out_mask.float()
            loss=loss_fn(mb_pred,mb_output,mb_out_mask)
            num_words=torch.sum(mb_y_len).item()
            total_loss+=loss.item()*num_words
            total_num_words+=num_words
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0) # 裁剪梯度
            optimizer.step()
            if it%100==0:
                print(f"Epoch:{epoch+1}/{num_epochs},Iter:{it}/{len(data)},Loss:{loss.item():.4f}")
            if epoch%5==0: # 每5轮进行评估
                evaluate(model,test_data)
        torch.save(model.state_dict(),f"./data/tmp/010.1.model.pth")


# 定义评估模型损失函数
def evaluate(model,data):
    """评估模型损失函数
    @param model: 模型
    @param data: 测试数据集
    """
    model.eval()
    total_num_words=total_loss=0. 
    with torch.no_grad():
        for it,(mb_x,mb_x_len,mb_y,mb_y_len ) in enumerate(data):
            mb_x=torch.from_numpy(mb_x).to(device).long()
            mb_x_len=torch.from_numpy(mb_x_len).to(device).long() 

            mb_input=torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_output=torch.from_numpy(mb_y[:,1:]).to(device).long()
            
            mb_y_len=torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0]=1
            mb_pred,attn=model(mb_x,mb_x_len,mb_input,mb_y_len)
            mb_out_mask=torch.arange(mb_y_len.max().item(),device=device)[None,:]<mb_y_len[:,None]
            mb_out_mask=mb_out_mask.float()
            loss=loss_fn(mb_pred,mb_output,mb_out_mask)
            num_words=torch.sum(mb_y_len).item()
            total_loss+=loss.item()*num_words
            total_num_words+=num_words
        print(f"损失评估，平均损失：{total_loss/total_num_words:.4f}")
    train(model,data,num_epoochs=20)

# 应用模型
def translate_dev(i:int):
    """翻译"""
    en_sent=" ".join([en_inv_dict[w] for w in en_test[i]])
    cn_sent=" ".join([cn_inv_dict[w] for w in cn_test[i]])
    print(f"原始英文句子：{en_sent}")
    print(f"原始中文句子：{cn_sent}"    )
    mb_x=torch.from_numpy(np.array(dev_en[i])).reshape(1,-1).to(device).long()
    mb_x_len=torch.from_numpy(np.array([len(dev_en[i])])).to(device).long()
    
    bos=torch.Tensor([[cn_dict["BOS"]]]).long().to(device)
    # 翻译
    translation,attn=model.translate(mb_x,mb_x_len,bos)
    translation=[cn_inv_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans=[]
    for word in translation:
        if word!="EOS":
            trans.append(word)
        else : 
            break
    cn_sent=" ".join(trans)
    print(f"翻译中文句子：{cn_sent}"    )

model.load_state_dict(torch.load("./data/tmp/010.1.model.pth"),map_location=device)
for i in range(1,5):
    translate_dev(i)
    print()


