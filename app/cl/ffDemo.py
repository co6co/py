import subprocess
import os
import cv2
from io import BytesIO
from ffmpeg import Progress
from ffmpeg import FFmpeg
import numpy as np
import time
import datetime
import asyncio
from co6co.task.thread import Executing
import signal
from co6co.utils import log
interrupted = False


source = 'rtsp://admin:lanbo8338299@192.168.3.200:554/Streaming/channels/101'
ffmpeg_path = 'G:\\ToolExe\\FFmpegs\\ffmpeg'
# 创建管道
r_pipe, w_pipe = os.pipe()
# 设置非阻塞模式
r_pipe = os.fdopen(r_pipe, 'rb', 0)
w_pipe = os.fdopen(w_pipe, 'wb', 0)


'''
process = subprocess.Popen([
    # "-i", "-" 使用管道
    # -f hls -hls_list_size 10 -hls_time 10 I:\hls\live\output.m3u8
    ffmpeg_path, "-i", "-",  "-f", "hls",   '-hls_time', '5', '-hls_list_size', '5', 'D:\\www\\live\\1111.m3u8'
], stdin=r_pipe)
'''


def generate_video_stream_src(duration=5, fps=25, width=640, height=480):
    """
    创建一个虚拟视频流
    """
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 编码器
    buffer = BytesIO()

    for _ in range(int(duration * fps)):
        # 生成一个随机帧
        frame = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

        # 将帧编码为H264
        success, encoded_frame = cv2.imencode('.jpg', frame)
        if success:
            buffer.write(encoded_frame.tobytes())

    buffer.seek(0)
    return buffer


async def generate_video_stream(width=640, height=480):
    while True:
        try:
            if (w_pipe.closed):
                log.info("w pipe closed.,退出线程.")
                break

            # 生成一个随机帧
            frame = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            font = cv2.FONT_HERSHEY_SIMPLEX
            now = datetime.datetime.now()
            cv2.putText(frame, now.strftime('%H:%M:%S'), (0, 50), font, 2, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, now.strftime('%Y-%m-%d'), (100, 0), font, 2, (0, 255, 0), 1, cv2.LINE_AA)

            # 将帧编码为H264
            success, encoded_frame = cv2.imencode('.jpg', frame)

            if success:
                data = encoded_frame.tobytes()
                w_pipe.write(data)
                w_pipe.flush()
        except Exception as e:
            # 出错大多为通道关闭
            log.info("执行出错", e)


'''
# 直播流 获取值
process = subprocess.Popen(
    ["streamlink",  "--stdout", "https://twitch.tv/zilioner", "best"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, text=True)
stdout, stderr = process.communicate()

# 获取返回码
returncode = process.returncode

# 输出结果
if returncode == 0:
    print("Streamlink started successfully:")
    print(stdout)
else:
    print("Error starting streamlink:")
    print(stderr)

# 检查是否有错误信息
if stderr:
    print("Error output:", stderr.strip())
'''
hls_options = {"f": 'hls', 'hls_time': '5', 'hls_list_size': '5'}
ffmpeg = (
    FFmpeg(executable=ffmpeg_path)
    .option("y")
    .input("pipe:0")
    # .input(source, rtsp_transport="tcp", rtsp_flags="prefer_tcp",)
    .output("D:\\www\\live\\1111.m3u8", **hls_options)
)

progress: Progress = None


@ffmpeg.on("progress")
def on_progress(p: Progress):
    global progress
    progress = p


# video_stream = generate_video_stream_src(20)
# ffmpeg.execute(video_stream.read())

async def exec():
    ffmpeg.execute(r_pipe)
    log.warn("ffmpeg退出。")

Executing("stream_theam", generate_video_stream)
Executing("ffmpeg_theam", exec)


def signal_handler(sig, frame):
    try:
        global interrupted
        print("Caught SIGINT signal.", interrupted, w_pipe.closed)
        if interrupted:
            return
        interrupted = True
        w_pipe.close()
        ffmpeg._process.terminate()
        r_pipe.close()
    except Exception as e:
        print(e)


signal.signal(signal.SIGINT, signal_handler)

while True:
    if interrupted:
        break
    time.sleep(1)
'''
while True:
    w_pipe.write(data)
    w_pipe.flush()
'''
