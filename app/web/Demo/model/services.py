import asyncio
from co6co.utils import log
import subprocess


class RTSPService:
    def __init__(self):
        self._active_processes = {}

    async def exec_ipconfig(self):
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *["ipconfig", "/all"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            err = stderr.decode('gbk')
            if err:
                log.err(f"ipconfig error: {err}") 
            yield stdout 

            #while True:
            #    data = await asyncio.wait_for(process.stdout.read(), timeout=2.0)
            #    if not data:
            #        break
            #    yield data
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


    async def read_out_stream(self,stdout: asyncio.StreamReader,chunk_size:int=64*1024): 
        while True:
            try: 
                data = await asyncio.wait_for(stdout.read(chunk_size), timeout=2.0)
                if not data: 
                    log.info("输出结束.")
                    break
                else:
                    yield data
            except Exception as e:
                log.err(f"读取数据异常: {e}")
                continue
    async def read_err_stream(self,stderr: asyncio.StreamReader,err_title:str="error"): 
        while True:
            try: 
                line = await stderr.readline()
                if not line:
                    break
                # 可选：记录错误日志
                print(f"{err_title}\t{line.decode().strip()}") 
            except Exception as e:
                log.err(f"{err_title}读取数据异常: {e}")
                continue
            

    async def read_rtsp_stream(self, rtsp_url: str, key: str ):
        """
        在多线程环境中，如果你在非主线程中使用asyncio，需要确保在该线程中有一个事件循环，并且异步函数在这个事件循环中运行
        
        假设我们在一个单独的线程中运行一个事件循环，然后在该事件循环中运行异步函数来创建子进程。
        步骤： 
        在新线程中创建事件循环。 
        在该事件循环中运行异步函数。 
        异步函数中创建子进程
        """
        # RTSP参数配置
        ffmpeg_cmd = [
            "E:/Tools/VXL/ffmpeg/bin/ffmpeg.exe",
            "-rtsp_transport",
            "tcp",  # 使用TCP传输
            "-i",
            rtsp_url,  # RTSP源
            "-f",
            "mp4",  # 输出格式
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
            "-c:v",
            "libx264",  # 视频编码
            "-preset",
            "ultrafast",  # 快速编码
            "-tune",
            "zerolatency",  # 零延迟
            "-b:v",
            "1000k",  # 比特率
            "-bufsize",
            "1000k",  # 缓冲区大小
            "-max_delay",
            "1000000",  # 最大延迟
            "-vsync",
            "1",  # 视频同步方式
            "-g",
            "10",  # 关键帧间隔
            "-reset_timestamps",
            "1",  # 重置时间戳
            "-vf",
            "fps=15,scale=640:480",  # 帧率和缩放
            "-f",
            "mp4",  # 输出格式
            "pipe:1",  # 输出到stdout
        ]
        print(f"[INFO] 开始RTSP流: {rtsp_url}")
        process = None
        try: 

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            print(f"[INFO] 1111111111111111")
            self._active_processes[key] = process 
            print(f"[INFO] 222222222222222222")
            # 如果能执行完成 使用下面
            ##stdout, stderr =await process.communicate()
            # 无限流或者很多
            # 同时读取stdout和stderr
            '''
            async def read_stream(stream:asyncio.streams.StreamReader, stream_name):
                # 逐行读取
                print(f"[INFO] 开始读取{type(stream)}")
                async for line in stream: 
                    if stream_name == 'STDOUT':
                        pass
                        #print("data", len( line))
                    else:
                        print(f'[{stream_name}] {line.decode()}', end='')
            await asyncio.gather(
                read_stream(process.stdout, 'STDOUT'),
                read_stream(process.stderr, 'STDERR')
            ) 
            # 等待进程结束
            #await process.wait() e_task(process.wait())
            ''' 
            #stderr_task = asyncio.create_task(self.read_err_stream(process.stderr,err_title="FFmpeg Error")) 
            
            # 读取并发送视频数据 
            chunk_size = 64 * 1024  # 64KB块 
            while True:
                try: 
                    data = await asyncio.wait_for(process.stdout.read(chunk_size), timeout=2.0)
                    if not data: 
                        log.info("输出结束.")
                        break
                    else:
                        yield data 
                except Exception as e:
                    log.err(f"读取数据异常: {e}")
                    continue  
        except Exception as e:
            print(f"[ERROR] 流异常: {e}")
            raise e
        finally:
            # 清理资源
            if process:
                try:
                    print(f"[INFO] 终止FFmpeg进程: {key}")
                    process.terminate()
                    log.info(f"等待FFmpeg进程退出: {key}")
                    await asyncio.wait_for(process.wait(), timeout=5)
                    log.info(f"FFmpeg进程退出: {key}")
                except Exception as e:
                    print(f"[INFO] 推出异常: {key}",e)
                    pass

            if key in self._active_processes:
                del self._active_processes[key]

            print(f"[INFO] 清理完成: {rtsp_url}")

    def stop(self, ws_key):
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
            self.stop(key)
