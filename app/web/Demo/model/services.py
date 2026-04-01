import asyncio
import subprocess 

class RTSPService:
    def __init__(self): 
        self._active_processes = {}
    
    async def read_rtsp_stream(self,rtsp_url: str,key:str):
        
        # RTSP参数配置
        ffmpeg_cmd = [
            'D:\\FTPHOME\\常用软件\\FFmpegs\\ffmpeg.exe',
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
            print(f"[INFO] 1111111111111111")
            self._active_processes[key] = process
            # 单独处理stderr（避免阻塞）
            async def read_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    # 可选：记录错误日志 
                    print(f"[FFmpeg Error] {line.decode().strip()}")
            print(f"[INFO] 222222222222222222")
            stderr_task = asyncio.create_task(read_stderr())
            print(f"[INFO] 333333333333333333")
        
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
            print(f"[ERROR] 流异常: {e}")  
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
            
            print(f"[INFO] 清理完成: {rtsp_url}") 
    def cleanup_process(self,ws_key):
        """清理FFmpeg进程"""
        if ws_key in self._active_processes:
            process = self._active_processes[ws_key]
            if process and process.returncode is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except:
                    process.kill()
            del self._active_processes[ws_key]
    def __del__(self):
        for key in self._active_processes.keys():
            self.cleanup_process(key)   
