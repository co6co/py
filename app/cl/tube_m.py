from pytubefix import YouTube
from pytubefix.cli import on_progress
import time
import os

def simple_audio_download(url, max_retries=5):
    """简单稳定的音频下载函数"""
    for retry in range(max_retries):
        try:
            print(f"\n尝试 {retry + 1}/{max_retries}")
            
            # 每次尝试都重新创建YouTube对象
            yt = YouTube(
                url, 
                on_progress_callback=on_progress,
                use_oauth=False,
                allow_oauth_cache=False
            )
            
            print(f"标题: {yt.title}")
            
            # 获取音频流
            audio_stream = yt.streams.get_audio_only()
            if not audio_stream:
                print("未找到音频流")
                return False
            
            print(f"下载音频: {audio_stream.abr}")
            
            # 设置较短的超时时间
            audio_stream.download(
                output_path="downloads",
                max_retries=3,
                timeout=30
            )
            
            print("下载成功！")
            return True
            
        except Exception as e:
            print(f"错误: {e}")
            if retry < max_retries - 1:
                wait = 2 ** retry
                print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)
            continue
    
    return False

# 主程序
url = input("请输入YouTube视频URL: ").strip()

# 创建下载目录
os.makedirs("downloads", exist_ok=True)

if simple_audio_download(url, max_retries=5):
    print("\n音频下载完成！")
else:
    print("\n多次尝试后下载失败，请检查：")
    print("1. 网络连接")
    print("2. URL是否正确")
    print("3. 视频是否可用")