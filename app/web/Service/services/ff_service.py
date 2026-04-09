import asyncio
import subprocess 
from co6co.utils import log

class ffService:
    def __init__(self): 
        self._active_processes = {}
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
    async def exec_ipconfig(self ):
        process =None
        try:
            process =  await asyncio.create_subprocess_exec(
                *['ipconfig','/all'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            while True:
                data = await asyncio.wait_for(
                            process.stdout.read(),
                            timeout=2.0
                        )
                if not data:
                    break
                yield data 
        except Exception as e:
            print(f"[ERROR] 执行异常: {e}")  
            raise e
        finally:
            # 清理资源
            if process:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except:
                    pass 


    async def read_rtsp_stream(self,rtsp_url: str,key:str): 
        # RTSP参数配置
        ffmpeg_cmd = [
            'ffmpeg.exe',
            '-rtsp_transport', 'tcp',          # 使用TCP传输
            '-i', rtsp_url,                    # RTSP源
            '-f', 'mp4',                       # 输出格式
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset',
            '-c:v', 'libx264',                 # 视频编码
            '-preset', 'ultrafast',           # 快速编码
            '-tune', 'zerolatency',           # 零延迟
            '-b:v', '1000k',                  # 比特率
            '-bufsize', '1000k',              # 缓冲区大小
            '-max_delay', '1000000',          # 最大延迟
            '-vsync', '1',                    # 视频同步方式
            '-g', '10',                       # 关键帧间隔
            '-reset_timestamps', '1',         # 重置时间戳
            '-vf', 'fps=15,scale=640:480',    # 帧率和缩放
            '-f', 'mp4',                      # 输出格式
            'pipe:1'                          # 输出到stdout
        ]
        print(f"[INFO] 开始RTSP流: {rtsp_url}")
        process=None
      
        try:   
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )   
            self._active_processes[key] = process
            # 单独处理stderr（避免阻塞）
            async def read_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    # 可选：记录错误日志 
                    print(f"[FFmpeg Error] {line.decode().strip()}") 
            stderr_task = asyncio.create_task(read_stderr())  
            # 读取并发送视频数据
            chunk_size = 64 * 1024  # 64KB块
            while True:
                try:
                    print(f"[INFO] read data")
                    data = await asyncio.wait_for(
                        process.stdout.read(chunk_size), 
                        timeout=2.0
                    )
                    
                    if not data:
                        print(f"[INFO] FFmpeg输出结束")
                        break 
                    else:
                        yield data
                except Exception as e: 
                    print(f"[ERROR] 读取数据异常: {e}")  
                    continue

        except Exception as e:
            print(f"[ERROR] 流生产器异常: {e}")  
            log.err(f"[ERROR] 流生产器异常: {e}")  
            raise e
        finally:
            # 清理资源
            if process:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except:
                    pass
            
            if key in self._active_processes:
                del self._active_processes[key]
                loop.close()
            
            print(f"[INFO] 清理完成: {rtsp_url}") 
     

    def stop(self,ws_key):
        """清理FFmpeg进程"""
        print(f"[INFO] 清理FFmpeg进程: {ws_key},{self._active_processes}")
        if ws_key in self._active_processes:
            process = self._active_processes[ws_key]
            print(f"[INFO] 收到 停止  清理FFmpeg进程: {ws_key}")
            if process and process.returncode is None:
                try:
                    print(f"[INFO] 调用 process.terminate(): {ws_key}")
                    process.terminate()
                    process.wait(timeout=3)
                except:
                    process.kill()
            del self._active_processes[ws_key]
        self._loop.close()
    def __del__(self):
        for key in self._active_processes.keys():
            self.stop(key)   
