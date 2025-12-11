#-*-coding:utf-8-*-
# 练习题
# PyTorch实现Word2Vec模型

import random,math,sys,time,os
from collections import Counter
from PIL.ImageFont import MAX_STRING_LENGTH
from nltk import corpus
from sympy import evaluate
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.qat.modules import embedding_ops
from torch.utils.data import DataLoader
from torch import nn,optim 
import torch.nn.functional as F

# 数据准备
corpus=['I like apple','He likes orange','She loves banana']
# 将文本语料转换为词汇表
corpus_tokens=[sentence.split() for sentence in corpus]
vocab=list( set([token for sentence in corpus_tokens for token in sentence]))
word_to_idx={word:i for i,word in enumerate(vocab)}
corpus_indices=[[word_to_idx[token] for token in sentence] for sentence in corpus_tokens]

# 超参数
embedding_dim=100
learning_rate=0.01
window_size=2
negative_samples=5
# 定义模型
class SkipGram(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super(SkipGram,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.linear=nn.Linear(embedding_dim,vocab_size)
    def forward(self,input_word):
        embeds=self.embedding(input_word)
        out=self.linear(embeds)
        
        return out

# 初始化模型和优化器
vocab_size=len(vocab)
model=SkipGram(vocab_size,embedding_dim)
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
loss_fn=nn.CrossEntropyLoss()

# 训练模型
for i,sentence in enumerate(corpus_indices):
    for j,center_word in enumerate(sentence):
        context_words=sentence[max(0,j-window_size):j]+sentence[j+1:min(j+window_size+1,len(sentence))]
        for context_word in context_words:
            # 正样本
            optimizer.zero_grad()
            center_word_var=Variable(torch.LongTensor([center_word]))
            context_word_var=Variable(torch.LongTensor([context_word]))
            output=model(center_word_var)
            loss=loss_fn(output,context_word_var)
            loss.backward()
            optimizer.step()

            # 负样本
            for _ in range(negative_samples):
                optimizer.zero_grad()
                random_word=np.random.randint(vocab_size)
                while random_word in context_words:
                    random_word=np.random.randint(vocab_size)
                random_word_var=Variable(torch.LongTensor([random_word]))
                output=model(center_word_var)
                loss=loss_fn(output,random_word_var)
                loss.backward()
                optimizer.step()

#应用模型
embedding=model.embedding.weight.data.numpy()
# 计算词语之间的相似度
def similarity(word1,word2):
    vec1=embedding[word_to_idx[word1]]
    vec2=embedding[word_to_idx[word2]]
    dot_product=np.dot(vec1,vec2)
    norm1=np.linalg.norm(vec1)
    norm2=np.linalg.norm(vec2)
    cosine_similarity=dot_product/(norm1*norm2)
    return cosine_similarity

def find_similar_words(word,k):
    """
    找到与给定词语最相似的k个词语   
    """
    similarities={}
    for w in vocab:
        if w!=word:
            similarities[w]=similarity(word,w)
    similar_word=sorted(similarities.items(),key=lambda x:x[1],reverse=True)[:k]
    return similar_word

# 示例
word='apple'
k=5
similar_words=find_similar_words(word,k)
print(f"与'{word}'最相似的{k}个词语:")
for w,sim in similar_words:
    print(f"{w}:{sim:.4f}")