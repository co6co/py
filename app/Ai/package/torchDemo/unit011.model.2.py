#-*-encoding=utf-8 -*-
# 建立模型示例 
import torch
from torch import nn,optim
import torch.utils.data as data

import torchaudio
from torchaudio import datasets,transforms
import soundfile
import matplotlib
import matplotlib.pyplot as plt
torchaudio.set_audio_backend("soundfile")
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
matplotlib.rcParams['axes.unicode_minus'] = False # 用于显示负号，避免出现乱码
dataset_path='./data/tmp/011.2'
dataset=datasets.DatasetFromFolder(root=dataset_path,loader=torchaudio.load)
dataloader=data.DataLoader(dataset,batch_size=64,shuffle=True)

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
       
        self.fc=nn.Linear(16*32*32,10)
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

model=AudioModel()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
    running_loss=0.0
    for inputs,labels in dataloader:
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    epoch_loss=running_loss/len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        