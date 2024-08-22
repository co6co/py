import threading
import asyncio
import time


async def background_task():

    print("Starting background task")
    await asyncio.sleep(5)

    print("Background task completed")


async def start():
    loop = asyncio.get_running_loop()  # loop=asyncio.get_event_loop() #已经被弃用，建议使用 get_running_loop
    print(loop)
    loop.create_task(background_task())
    print("main.stoped.")
    time.sleep(500)

##########################################################


async def my_coroutine(name: str):
    print("Coroutine '{}' started".format(name))
    await asyncio.sleep(1)
    print("Coroutine '{}' finished".format(name))


def run_coroutine_in_new_loop(name: str):
    # 创建一个新的事件循环
    #
    # 程序已经有一个默认的事件循环，
    # 通常不需要手动创建新的事件循环。
    # 使用 asyncio.get_event_loop() 或 asyncio.get_running_loop() 即可。

    # 多线程环境中使用 asyncio.new_event_loop(), 确保每个线程都有自己的事件循环
    # 并且不要在不同的线程间共享同一个事件循环。

    loop = asyncio.new_event_loop()
    print("loop created")
    try:
        # 设置当前线程的事件循环
        asyncio.set_event_loop(loop)

        # 在新的事件循环中运行协程
        loop.run_until_complete(my_coroutine(name))  # run_until_complete 方法来运行协程
    finally:
        # 清理事件循环
        print("loop close")
        loop.close()


def run_coroutine_in_thread():
    run_coroutine_in_new_loop("线程+newLoop")


###########################################################
if __name__ == '__main__':
    print("main start .......")
    run_coroutine_in_new_loop("newLoop")

    thread = threading.Thread(target=run_coroutine_in_thread)
    thread.start()
    print("main end.")
