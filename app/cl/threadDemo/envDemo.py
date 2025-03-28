
import asyncio


async def main():
    """ 
    主函数，用于创建并运行异步任务。
    该函数创建了两个异步任务：waiter 和 setter，分别用于等待事件和设置事件。
    """
    async def waiter(event: asyncio.Event, lock: asyncio.Lock, name: str):
        while True:
            async with lock:  # 加锁
                await event.wait()  # 等待事件
                print(f"{name} 开始执行")
                # await asyncio.sleep(1)  # 等待 1 秒,不然当event.wait() 不用等待时其它协程得不到执行
                event.clear()  # 重置事件 否则 event.wait 不用等待

    async def setter(event: asyncio.Event):
        while True:
            await asyncio.sleep(1)  # 每秒设置一次事件
            print("设置 Event set!")
            if not event.is_set():
                event.set()
            else:
                print("event is set")

    event = asyncio.Event()
    lock = asyncio.Lock()
    # 创建任务
    waiter_tasks = [asyncio.create_task(waiter(event, lock, f"协程 {i}")) for i in range(3)]
    setter_task = asyncio.create_task(setter(event))
    # 等待所有任务完成
    await asyncio.gather(*waiter_tasks, setter_task)

if __name__ == '__main__':
    asyncio.run(main())
