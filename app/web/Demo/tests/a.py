import asyncio,json
from typing import AsyncGenerator
from model.services import RTSPService
from model.apphelp import get_config

# 示例 1: 基础的异步生成器
async def get_data() -> AsyncGenerator[str, None]:
    """返回字符串数据的异步生成器"""
    for i in range(5):
        # 模拟异步操作
        await asyncio.sleep(0.5)
        yield f"数据块 {i}\n"

async def get_json_data() -> AsyncGenerator[str, None]:
    """返回JSON格式数据的异步生成器"""
    for i in range(3):
        await asyncio.sleep(0.1)
        data = {
            "id": i,
            "message": f"消息 {i}",
            "timestamp": asyncio.get_event_loop().time()
        }
        yield json.dumps(data) + "\n"

# 示例 3: 模拟从 RTSP 流读取数据的异步生成器
async def get_rtsp_stream_data(rtsp_url: str) -> AsyncGenerator[bytes, None]:
    """从RTSP流读取数据的异步生成器"""
    import subprocess
    
    # 创建FFmpeg进程
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-f', 'mp4',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'mp4',
        'pipe:1'
    ]
    
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        chunk_size = 64 * 1024  # 64KB
        
        while True:
            # 从进程读取数据
            chunk = await process.stdout.read(chunk_size)
            if not chunk:
                break
            
            # 生成数据块
            yield chunk
            
    except asyncio.CancelledError:
        print("流被取消")
        raise
    except Exception as e:
        print(f"读取流时出错: {e}")
    finally:
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
async def async_generator():
    for i in range(5):
        await asyncio.sleep(0.5)  # 模拟异步操作
        yield i

async def generator_expression():
    agen = (x * 2 async for x in async_generator())
    async for item in agen:
        print(item)  # 0, 2, 4, 6, 8

if __name__ == "__main__":
   
    async def test(): 
        config=get_config()
        url=config.get("rtsp_url")
        test2= get_rtsp_stream_data(url)
        async for data in test2:
            print(data)
    asyncio.run(test())
   