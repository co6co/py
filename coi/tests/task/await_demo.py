import asyncio
from math import factorial
from co6co.utils import log
from typing import AsyncIterator
import aiofiles
import os
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_file_lock(filename: str, mode: str = "r"):
    """异步文件锁，确保同一时间只有一个协程访问文件"""
    lock_file = f"{filename}.lock"
    
    # 等待获取锁
    while os.path.exists(lock_file):
        print(f"等待文件锁: {lock_file}")
        await asyncio.sleep(0.1)
    
    # 创建锁文件
    async with aiofiles.open(lock_file, "w") as f:
        await f.write("locked")
    
    try:
        # 打开目标文件
        async with aiofiles.open(filename, mode) as file:
            print(f"获取文件锁: {filename}")
            yield file
    finally:
        # 释放锁
        os.remove(lock_file)
        print(f"释放文件锁: {filename}")

# 使用
async def write_with_lock():
    async with async_file_lock("data.txt", "a") as file:
        await file.write(f"数据写入 at {asyncio.get_event_loop().time()}\n")
        await asyncio.sleep(1)  # 模拟耗时操作
        print("文件写入完成")

async def test_concurrent_writes():
    """测试并发写入，可以看到锁的作用"""
    tasks = [write_with_lock() for _ in range(3)]
    await asyncio.gather(*tasks)


async def do_something(msg:str=''):
     await asyncio.sleep(1)
      
     print(msg)
     # 模拟异步操作
     return 42

async def main():
     log.start_mark("单任务")
     result = await do_something()  # 等待do_something协程完成，并获取返回值
     print(result)
     log.end_mark("单任务")

async def main2():
     log.start_mark("多任务")
     task1 = asyncio.create_task(do_something('start'))
     task2 = asyncio.create_task(do_something('hello'))
     # 等待两个任务都完成
     print(await task1)
     print(await task2)
     # 或者使用asyncio.gather
     results = await asyncio.gather(task1, task2)
     print(results)
     log.end_mark("多任务")

class AsyncDataStream:
    """异步数据流迭代器""" 
    def __init__(self, data_source: list[str], delay: float = 0.1):
        self.data = data_source
        self.delay = delay
        self.index = 0
    
    def __aiter__(self) -> AsyncIterator[str]:
        return self
    
    async def __anext__(self) -> str:
        """返回下一个元素"""
        if self.index >= len(self.data):
            raise StopAsyncIteration
        
        # 模拟异步获取数据
        await asyncio.sleep(self.delay)
        item = self.data[self.index]
        self.index += 1
        
        return item
    
    async def reset(self):
        """重置迭代器"""
        self.index = 0

async def async_generator():
    for i in range(100):
        yield i     
async def process(item):
    log.info(item)
async def must_use_await():
    """
    必须使用 await 的情况
    """
    # 1. 调用其他异步函数
    await asyncio.sleep(1)
    
    # 2. 读取异步流
    reader, writer = await asyncio.open_connection('python.org', 80)
    data = await reader.read(100)  # await
    writer.close()
    await writer.wait_closed()  # await
    
    # 3. 执行子进程
    proc = await asyncio.create_subprocess_exec('ls')
    await proc.wait()  # await
    
    # 4. 等待 Future/Task
    future = asyncio.Future()
    await future  # 等待 future 完成
    
    # 5. 使用异步上下文管理器
    async with async_file_lock('data.txt','r') as f:
        data = await f.read()
        print(data)
    
    # 6. 使用异步迭代器
    async for item in AsyncDataStream(['a', 'b', 'c']):  # 内部使用 await
        await process(item)  # await
    
    # 7. 使用 anext() 和 aiter()
    async_gen = async_generator()
    value = await anext(async_gen)  # await

async def no_await_needed():
    """
    不需要使用await 的情况
    """
    # 1. 创建任务（不等待）
    task = asyncio.create_task(asyncio.sleep(1))
    # 这里不 await，任务在后台运行
    
    # 2. 同步函数调用
    result = len([1, 2, 3])  # 同步，不用 await
    print(result)  # 同步，不用 await
    
    # 3. 同步上下文管理器
    with open('file.txt') as f:
        content = f.read()  # 同步，不用 await
    
    # 4. 同步迭代
    for i in range(10):  # 同步，不用 await
        print(i)
    
    # 5. 返回普通值
    return 42  # 不用 await
if __name__ == '__main__':
    asyncio.run(main())
    asyncio.run(main2())