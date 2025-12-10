#-*- coding: utf-8 -*-
# 文本
# Word2Vec提取相似文本
import random,math,sys,time
import torch
from torch.utils.data import DataLoader
from torch import nn,optim
# 加载数据集
with open('./data/tmp/010.1/HarryPotter.txt','r',encoding='utf-8') as f:
    reviews=f.readlines()
    # 对列表中的每一行，使用Split()方法将其分割成单词，
    # 并将结果存储在一个新的列表中
    raw_dataset=[review.split() for review in reviews]
# 为了计算简单，我们只保留在数据集中至少出现5次的次，然后将词映射到整数索引
counter=collections.Counter([word for review in raw_dataset for word in review])
# 过滤出现次数大于等于5的单词
counter=dict(filter(lambda x:x[1]>=5,counter.items()))
# 提取所有出现次数大于等于5的单词
idx_to_token=[tk for tk,_ in counter.items()]
# 建立单词到索引的映射
token_to_idx={tk:idx for idx,tk in enumerate(idx_to_token)}
# 将原始数据集转化为索引数据集
dataset=[[token_to_idx[tk] for tk in review if tk in token_to_idx] for review in raw_dataset]
# 计算数据集中的单词数
num_tokens=len([len(st) for st in dataset])
# 文本中一般会出现一些高频词，如英文中的the，a和in，通常来说，在一个北京窗口中，一个词和较低频词同时出现闭合较高频词同时出现对训练词嵌入模型更有益。
# 因此训练词嵌入模型时可以对词进行二次采样。
def discard(idx):
    return random.uniform(0,1)<1-math.sqrt(1e-4/counter[idx_to_token[idx]]*num_tokens)
subsampled_dataset=[[tk for tk in st if not discard(tk)] for st in dataset]
# 提取中心词和背景词，将与中心词距离不超过背景窗口大小的词作为它的背景词。
# 下面定义函数提取出所有中心词和他们的背景词，它每次在整数1和max_windows_size(最大背景窗口)之间随机均匀采样一个整数作为背景窗口大小
def get_centers_and_contexts(dataset,max_window_size):
    """
    目的：从给定的数据集dataset中获取中心和上下文
    参数：
        dataset：输入的数据集，每个元素是一个词的索引列表
        max_window_size：最大背景窗口大小
    返回：
        (centers：中心词列表
        contexts：上下文词列表
        )
    """
    centers,contexts= [],[]
    for st in dataset:
        if len(st)<2:
            continue
        #centers+=[tk for tk in st]
        centers.append(st)
        for center_idx in range(len(st)):
            window_size=random.randint(1,max_window_size)
            indices=list(range(max(0,center_idx-window_size),min(len(st),center_idx+window_size+1)))
            indices.remove(center_idx)
            contexts.append([st[idx] for idx in indices])
    return centers,contexts
# 我们假设最大北京窗口大小为5，下面提取数据集中所有的中心词及其背景词
# 存储通过 get_centers_and_contexts 函数获取到的结果
all_centers,all_contexts=get_centers_and_contexts(subsampled_dataset,5)
# 使用负采样来进行近似训练，对于一个中心词和背景词，我们随机采样K个噪声词（这里设置为5），根据Word2V2c论文建议
# 噪声词采样P(w)设置w词频与总词频之比的0.75次方
def get_negatives(all_contexts,sampling_weights,K):
    """
    目的：根据上下文词列表和语料库词频统计，为每个上下文词采样K个噪声词
    参数：
        all_contexts：上下文词列表，每个元素是一个上下文词的索引列表
        sampling_weights
        K：每个上下文词采样的噪声词数量
    返回：
        negatives：噪声词列表，每个元素是一个噪声词的索引列表
    """
    all_negatives,neg_candidates,i=[],[],0
    # 生成一个包含采样权重对应索引的列表
    population=list(range(len(sampling_weights)))
    for contents in all_contexts:
        negatives=[]
        while len(negatives)<len(contents)*K:
             
            if i==len(neg_candidates):
                i,neg_candidates=0,random.choices(population,weights=sampling_weights,k=10000) 
            neg,i=neg_candidates[i],i+1 
            if neg not in set(contents):
                negatives.append(neg)
            all_negatives.append(negatives)
    return all_negatives
# 计算采用权重，使用计数器中每个元素的0.75次方
sampling_weights=[counter[tk]**0.75 for tk in idx_to_token]
# 调用get_negatives函数，为每个上下文词采样5个噪声词
all_negatives=get_negatives(all_contexts,sampling_weights,5)

# 小批量读取函数 batchify
# 目的：将中心词、上下文词和噪声词列表打包成小批量
# 参数：
#     centers：中心词列表，每个元素是一个中心词的索引列表
#     contexts：上下文词列表，每个元素是一个上下文词的索引列表
#     negatives：噪声词列表，每个元素是一个噪声词的索引列表
# 返回：
#     (centers：中心词小批量
#     contexts：上下文词小批量
#     negatives：噪声词小批量
#     )
def batchify(data):
    """
    目的：对数据进行批处理
    参数：
        data：list 包含中心、上下文和负样本数据
         
    返回：
         批处理后的数据，包括中心、上下文和负样本、掩码和标签
    """
    max_len=max(len(c)+len(n) for _,c,n in data)
    centers,contexts_negatives,masks,labels=[],[],[],[],[]
    for center,context,negative in data:
        cur_len=len(context)+len(negative)
        centers+=[center]
        contexts_negatives+=[context+negative+[0]*(max_len-cur_len)]
        masks+=[[1]*cur_len+[0]*(max_len-cur_len)]
         
        labels+=[[1]*len(context)+[0]*(max_len-len(context))]
        batch=(torch.tensor(centers).view(-1,1),
        torch.tensor(contexts_negatives),
        torch.tensor(masks),
        torch.tensor(labels)
        )
        return batch
 
# DataLoader 读取
batch_size=256
num_workers=0 if sys.platform.startswith('win32') else -1
dataset=MyDataset(all_centers,all_contexts,all_negatives)
data_iter=DataLoader(dataset,batch_size,shuffle=True,collate_fn=batchify,num_workers=num_workers)
for batch in data_iter:
    for name,data in zip(['centers','contexts_negatives','masks','labels'],batch):
        print(name,"shape",data.shape   )
        break

# 网络模型
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss,self).__init__() 
    def forward(self,inputs,targets,masks=None):
        """
        将输入、目标和掩码转换为浮点型
        """
        inputs,targets,masks=inputs.float(),targets.float(),masks.float()
        # 使用nn.functional.binary_cross_entropy_with_logits计算二进制交叉熵损失
        res=nn.functional.binary_cross_entropy_with_logits(inputs,targets,reduction='none',weight=masks)
        res=res.sum(dim=-1)/masks.sum(dim=1)
        return res 
loss=SigmoidBinaryCrossEntropyLoss()
def sigmoid(x):
    return -math.log(1/(1+math.exp(-x)))

# 定义网络
embed_size=200 # 嵌入向量大小200
net=nn.Sequential(
    # 嵌入层 将输入的索引映射到输入的向量，
    nn.Embedding(num_embeddings=len(idx_to_token),embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token),embedding_dim=embed_size)
)

# 测试
def skip_gram(center,contexts_and_negatives,embed_v,embed_u):
    """
    计算skip-gram模型的预测
    参数：
        center：中心词索引
        contexts_and_negatives：上下文词和噪声词索引列表
        embed_v：输入映射到嵌入向量的函数
        embed_v：输入映射到嵌入向量的函数
    返回：
        pred：torchTensor预测值
    """
    v=embed_v(center) #中心词映射到嵌入向量v
    u=embed_u(contexts_and_negatives).permute(0,2,1)# 将上下文词和负样本映射到嵌入向量u并进行转置
    pred=torch.bmm(v,u) # 计算v和转置后的u的批量矩阵乘法
    return pred
def train(net,lr,num_epochs):
    """
    训练skip-gram模型
    参数：
        net：skip-gram模型网络
        lr：学习率
        num_epochs：训练轮数
    """
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    optimizer=optim.Adam(net.parameters(),lr=lr)
    for epoch in range(num_epochs):
        start,l_sum,n=time.time(),0.0,0
        for i,batch in enumerate(data_iter):
            center,contexts_negatives,masks,labels=[data.to(device) for data in batch]
            pred=skip_gram(center,contexts_negatives,net[0],net[1])
            l=loss(pred.view(labels.shape),labels,masks).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum+=l.cpu().item()
            n+=1
            if i%2000==1999:
                print('[%d,%5d] loss: %.3f,time:%.3f'%(epoch+1,i+1,l_sum/n,time.time()-start))
train(net,0.01,5)

# 应用网络模型
def get_similar_tokens(query_token,k,embed):
    """
    获取与查询词最相似的k个词
    参数：
        query_token：查询词
        k：返回的相似词数量
        embed：嵌入层
    返回：
        tokens：与查询词最相似的k个词列表
        sims：与查询词最相似的k个词的相似度列表
    """
    W=embed.weight.data
    x=W[idx_to_token[query_token]]
    # 计算余弦相似度
    cos=torch.matmul(W,x)/(torch.sqrt(torch.sum(W*W,dim=1))*torch.sqrt((x*x).sum()))
    _,topk=torch.topk(cos,k=k+1)
    topk=topk.cpu().numpy().tolist()
    for i in topk[1:]:
        print("余弦相似度:",cos[i].item(),"词:",idx_to_token[i])
    #tokens=[idx_to_token[i] for i in topk[1:]]
    #sims=[cos[i].item() for i in topk[1:]]
    #return tokens,sims

get_similar_tokens('Dursley',10,net[0])