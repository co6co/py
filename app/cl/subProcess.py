import subprocess
import numpy as np
import cv2
import datetime
import time
import io
# from typing_extensions import IO
from typing import IO
from co6co.utils import log, read_stream, write_stream
from co6co.task.thread import Executing
ffmpeg_path = 'G:\\ToolExe\\FFmpegs\\ffmpeg'
hls_options = {"f": 'hls', 'hls_time': '10', 'hls_list_size': '10'}


async def generate_video_stream(width=640, height=480, io: IO[bytes] = None):
    while True:
        try:
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
                #print("wirteData:", len(data))

                io.write(data)
                io.flush()
            #time.sleep(0.1)
        except Exception as e:
            # 出错大多为通道关闭
            log.info("执行出错", e)
            break


async def quitCheck(process: subprocess.Popen):
    # process.wait()
    # 子线程是否退出
    # exited = process.poll() is not None
    log.warn("等待ff 子线程退出..") 
    stdout, stderr = process.communicate()
    log.warn("等待ff 子线程退出.")
    # 检查子进程的退出码
    exit_code = process.returncode
    if exit_code != 0:
        print(f"子进程退出码：{exit_code}")
        print(f"标准错误输出：{stderr}")

    print(f"标准输出：{stdout}")
    
    #poll() 返回退出状态码；否则返回 None
    '''
    while proc.poll() is None:
        print("Process is still running...")
    print(f"Process exited with return code: {proc.returncode}")
    '''


async def outPut(outIO: IO[bytes] = None):
    for chunk in read_stream(outIO, size=512):
        log.info(chunk)


async def errorPut(outIO: IO[bytes] = None):
    for chunk in read_stream(outIO, size=512):
        log.warn(chunk)

flv={"-f":"flv","-c": "copy"}
hls={"-f":"hls","-hls_time":"5","-hls_list_size":"5","-c":"copy"}
sizeArg={"-c:v": "libx264", "-b:v": "1000k", "-maxrate":"1500k", "-bufsize": "1500k", "-c:a": "aac", "-b:a": "128k",}
speed={"-bufsize":"1M","-preset":"ultrafast","-threads":"4"}
allArg={}
allArg.update(flv)
#allArg.update(sizeArg)
#allArg.update(speed) 
array=[]
[array.extend([t,allArg.get(t)]) for t in allArg] 
 

process = subprocess.Popen([
    # "-i", "-" 使用管道 
    #"-re" ,'-y', 
    # -f hls -hls_list_size 10 -hls_time 10  "D:\www\live\\1111.m3u8" 
    ffmpeg_path, "-i", "rtsp://admin:lanbo8338299@192.168.3.200:554/Streaming/channels/101", *array, "rtmp://192.168.1.99:1935/live/1111"
], stdin=None, stdout=None, stderr=None)
'''
process=subprocess.Popen(ffmpeg_path+" -i rtsp://admin:lanbo8338299@192.168.3.200:554/Streaming/channels/101 -f hls -hls_time 10 -hls_list_size 5 -c copy D:\www\live\\2222.m3u8", stdin=None, stdout=None, stderr=None)
'''
# Executing("quit", quitCheck, process)
Executing("stream", generate_video_stream, io=process.stdin)
#Executing("outStream", outPut, outIO=process.stdout)
#Executing("errStream", errorPut, outIO=process.stderr) 
# quitCheck()
# 等待子程序退出

log.warn('end等待进程退出')
process.wait()
log.warn('End进程退出')
