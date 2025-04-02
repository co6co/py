import asyncio


async def long_running_task():
    print("任务以开始..需要5秒.才能完成任务....")
    await asyncio.sleep(5)
    print("任务完成【设计不会执行到】.")


async def stop_loop_after(loop, delay):
    print(f"在 {delay} 秒后停止事件循环.")
    await asyncio.sleep(delay)
    loop.stop()
    print("loop被停止.")

loop = asyncio.get_event_loop()
# 创建长时间运行的任务
task = loop.create_task(long_running_task())
# 创建一个任务来在2秒后停止事件循环
stop_task = loop.create_task(stop_loop_after(loop, 4.95))

try:
    # 让事件循环一直运行，直到调用 loop.stop() 方法。
    # 在循环运行期间，事件循环会不断地检查是否有新的任务需要执行，并处理各种异步事件。
    #
    # 适用于需要持续运行的服务，例如网络服务器，它需要不断地接受新的连接和处理请求。
    loop.run_forever()
    print("事件循环已停止.")

    # 运行事件循环，直到传入的 future 对象（可以是协程、任务或 Future 对象）完成。
    # 一旦 future 完成，事件循环就会停止，并返回 future 的结果。

    # 适用于一次性执行的异步任务，例如执行一个异步函数并获取其结果。
    # loop.run_until_complete(future)

    # 将一个协程包装成一个 Task 对象，并将其添加到事件循环中。
    # Task 是 Future 的子类，它表示一个正在运行的协程。
    # loop.create_task(coro)
    # t1 = loop.create_task(task1()) # async def task1()..
    # t2 = loop.create_task(task2())
    # loop.run_until_complete(asyncio.gather(t1, t2))
except KeyboardInterrupt:
    pass
finally:
    # 确保任务被取消
    task.cancel()
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    # loop.close()方法用于关闭事件循环。一旦调用了close()方法，事件循环将不再接受新的任务，并且会释放所有相关的资源，如文件描述符、线程等。
    # close()方法只能在事件循环停止后调用，如果在事件循环正在运行时调用close()，会引发RuntimeError。
    loop.close()


# close
print("*** close ***"*10)


async def running_task():
    print("任务开始...")
    await asyncio.sleep(2)
    print("任务完成.")
loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(running_task())
finally:
    # 确保事件循环停止后再关闭
    print("Loop is running:", loop.is_running())
    if loop.is_running():
        loop.stop()
    loop.close()
    print("Loop closed")
