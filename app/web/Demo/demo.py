import asyncio
import threading
import sys
from model.services import RTSPService
async def run_subprocess():
    # 创建子进程
    #process = await asyncio.create_subprocess_exec(
    #    sys.executable, '-c', 'print("Hello from subprocess")',
    #    stdout=asyncio.subprocess.PIPE
    #)
    #
    ## 读取输出
    #stdout, _ = await process.communicate()
    #print(f"子进程输出: {stdout.decode().strip()}")
    #return stdout
    service = RTSPService()
    test2 = service.read_rtsp_stream(
        "rtsp://admin:lanbo12345@192.168.3.1/media/video1", "123"
    )
    async for data in test2:
        print(data )

def thread_main(loop: asyncio.AbstractEventLoop=None):
    # 设置该线程的事件循环
    #asyncio.set_event_loop(loop)
    # 运行异步函数
    if loop is None:
        loop = asyncio.get_event_loop()
        #asyncio.set_event_loop(loop)
    loop.run_until_complete(run_subprocess())

def main():
    # 创建新的事件循环
    #new_loop = asyncio.new_event_loop()
    # 创建线程，并传入事件循环
    thread = threading.Thread(target=thread_main, args=())
    thread.start()
    thread.join()

if __name__ == "__main__":
    main()