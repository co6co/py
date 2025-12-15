#-*-encoding=utf-8 -*-
# 练习
# 音频相似度分析
# 余弦相似度
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
filename='./data/tmp/011/1.mp3'
waveform1,sample_rate1=torchaudio.load(filename)
plt.figure()
plt.plot(waveform1.t().numpy())
plt.show()
waveform2,sample_rate2=torchaudio.load(filename)

# 相似度
similarity=torch.cosine_similarity(waveform1,waveform2,dim=0)
print("打印相似度",similarity.mean())
