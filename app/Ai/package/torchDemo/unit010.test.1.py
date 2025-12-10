#-*- coding: utf-8 -*-
# 机器翻译
# 使用Seq2Seq模型
# 数据集分为训练集和测试集两个文件
import random,math,sys,time,os
from collections import Counter
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
#....未完成。