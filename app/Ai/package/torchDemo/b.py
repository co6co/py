
import av
import whisper
import numpy as np
import os
import sys
import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

# 禁用tqdm进度条的多种方法
os.environ["TQDM_DISABLE"] = "1"
os.environ["NO_PROGRESS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 尝试直接禁用tqdm
from tqdm import tqdm as original_tqdm
def tqdm_disabled(*args, **kwargs):
    # 确保disable参数只被设置为True，不重复传递
    kwargs['disable'] = True
    return original_tqdm(*args, **kwargs)

import tqdm
tqdm.tqdm = tqdm_disabled

# 初始化whisper模型
model = whisper.load_model("base")

#url='rtsp://192.168.3.12/media/video1'
url='D:\\BaiduNetdiskDownload\\2018年 互联网技术 视频课件\\2018中级综合能力精讲视频\\第三章 计算机应用基础（2）..mp4'
container = av.open(url)

# 音频帧缓冲区
audio_buffer = []

for packet in container.demux():
    if packet.stream.type == 'audio':
        for frame in packet.decode():
            # 获取音频数据
            audio_data = frame.to_ndarray()
            
            # 将多声道转换为单声道（如果需要）
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # 检查音频数据是否为零
            if np.max(np.abs(audio_data)) < 0.01:
                continue
            
            # 添加到缓冲区
            audio_buffer.append(audio_data)
            
            # 当缓冲区达到一定长度时进行识别
            if len(audio_buffer) > 20:
                # 合并音频数据
                combined_audio = np.concatenate(audio_buffer)
                
                # 确保采样率为16kHz（whisper要求）
                sample_rate = frame.sample_rate
                
                # 处理采样率，确保步长不为零
                if sample_rate == 0:
                    sample_rate = 16000
                
                resampled_audio = combined_audio
                if sample_rate != 16000:
                    # 计算重采样步长
                    step = sample_rate // 16000
                    if step == 0:
                        step = 1
                    # 这里简单处理，实际应用中应该使用重采样
                    resampled_audio = combined_audio[::step]
                
                # 将音频数据转换为float32类型，匹配Whisper模型的期望
                resampled_audio = resampled_audio.astype(np.float32)
                
                # 归一化音频数据到[-1, 1]范围
                max_val = np.max(np.abs(resampled_audio))
                if max_val > 0:
                    resampled_audio = resampled_audio / max_val
                
                try:
                    # 进行语音识别，禁用详细输出
                    result = model.transcribe(resampled_audio, language="zh", verbose=False)
                    
                    # 输出识别结果
                    if result["text"].strip():
                        print("内容",result["text"])
                except Exception as e:
                    print(f"识别错误: {e}")
                    pass
                
                # 清空缓冲区
                audio_buffer = []