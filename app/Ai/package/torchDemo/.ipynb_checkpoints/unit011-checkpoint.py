# -*- coding: utf-8 -*-
# 音频处理
# 需按照 PyTorch的torchaudio库和soundfile库
# torchaudiio.set_audio_backend() linux/macOs 使用 sox_io 、windows 使用SoundFile
import torch
import torchaudio
import soundfile
import matplotlib
import matplotlib.pyplot as plt
torchaudio.set_audio_backend("soundfile")
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
matplotlib.rcParams['axes.unicode_minus'] = False # 用于显示负号，避免出现乱码
# torchaudio 支持以WAV和MP3格式加载声音文件，我们称波形为原始音频信号。
filename='./data/tmp/011/1.mp3'
waveform,sample_rate=torchaudio.load(filename)

# 1. 波形图
print(f"波形形状 Waveform shape:{waveform.size()}")
print(f"波形采样率Sample rate:{sample_rate}")
plt.figure() # 创建一个图形
plt.plot(waveform.t().numpy()) # 绘制波形的时间序列
plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show(   )

# 2. 绘制波形频谱图
# 使用torchaudio库中的Spectrogram函数将波形数据转换为频谱图
specgram=torchaudio.transforms.Spectrogram()(waveform)
print(f"频谱形状(频谱图的尺寸):{specgram.size()}")
plt.figure() # 新窗口
#显示频谱的对数变换结果
# specgram.log2()[0,:,:].numpy()表示取频谱图的对数变换结果，并将其转换为numpy数组
# cmap=gray 用于指定颜色映射为灰度图
# aspect=auto 自动调整图像横纵比
plt.title("波形频谱图")
plt.imshow(specgram.log2()[0,:,:].numpy(),cmap='gray',aspect='auto')
plt.show()

# 3. 梅尔频谱图
# 以对数刻度查看梅尔频谱图
# 使用 MelSpectrogram 函数对波形进行梅尔频谱图变换
# 可以观察信号在梅尔频谱制度上的能量分布情况
# 梅尔频谱图是一种图书的频谱表示，他基于梅尔频率尺度，常用于语音处理等领域
specgram=torchaudio.transforms.MelSpectrogram()(waveform)
print("梅尔频谱图形状：",specgram.size())
plt.figure() 
plt.title("梅尔频谱图")
p=plt.imshow(specgram.log2()[0,:,:].detach().numpy(),cmap='viridis',aspect='auto')
plt.show()

# 3.1 重新采样
## 重新采样
# 一次一个通道
new_sample_rate=sample_rate /15 #
# 荀子要处理的通道，
channel=0
# 使用Resample函数对不行进行重新采样，
# 将波形数据的指定通道转换为一维张量
transformed=torchaudio.transforms.Resample(sample_rate,new_sample_rate)(waveform[channel,:].view(1,-1))
print(f"重新采样后的形状：{transformed.size()}")
plt.figure()
plt.plot(transformed[0,:].numpy()) 
plt.title("重新采样后的波形")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# 4. 波形的 Mu-Law编码 解码
# 4.1 编码
# Mu_Law 与反Mu_Law变换

def normalize(tensor:torch.Tensor)->torch.Tensor:
    """
    对输入的张量进行归一化处理
    @param tensor  :需要归一化的张量
    返回
    normalized_tensor  :归一化后的张量
    """
    tensor_minusmean=tensor-tensor.mean()
    # 减去均值后除以其绝对值的最大值
    return tensor_minusmean/tensor_minusmean.abs().max()

waveform_=normalize(waveform)
transformed=torchaudio.transforms.MuLawEncoding()(waveform_)
print("变换后波形形状：",transformed.size())
plt.figure()
plt.plot(transformed[0,:].numpy()) 
plt.title("Mu-Law 编码后的波形")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# 4.1 解码
reconstructed=torchaudio.transforms.MuLawDecoding()(transformed)
print("解码后波形形状：",reconstructed.size())
plt.figure()
plt.plot(reconstructed[0,:].numpy()) 
plt.title("Mu-Law 解码后的波形")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# 计算原始波形和重构波形之间的差异
err=((waveform-reconstructed).abs()/waveform.abs()).mean()
print("原始波形与重构之间的差异:",f"{err:.2%}")