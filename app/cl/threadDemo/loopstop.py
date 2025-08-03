import asyncio


async def long_running_task():
    print("任务以开始..需要5秒.才能完成任务....")
    await asyncio.sleep(5)
    print("任务完成【设计不会执行到】.")


async def stop_loop_after(loop, delay):
    """
    ### 1. 事件循环的作用
    事件循环是 asyncio 的核心，负责调度和执行所有异步任务。当你调用 loop.close() 后，事件循环会被完全关闭，不再处理任何新的或正在运行的任务 9 。
    loop.stop() 的非终止性 : loop.stop() 只是停止事件循环的调度，不会立即中断正在执行的任务。已经开始执行的协程会继续执行直到完成或遇到下一个 await 点。
    ### 2. 未取消任务的状态
    - 当事件循环关闭时，任何正在运行的任务都会被 立即中断
    - 这些任务不会继续执行，因为没有事件循环来调度它们
    - 任务的状态会变为未完成，但由于事件循环已关闭，它们永远不会被完成
    ### 3. 为什么需要调用 task.cancel()
    - 虽然事件循环关闭后任务会被中断，但调用 task.cancel() 是一种良好的实践
    - 它允许任务有机会捕获 asyncio.CancelledError 异常并进行资源清理
    - 这可以防止资源泄漏或任务处于不一致状态
    """
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
    #task.cancel() # 如果不调用 task可能会完成【在这里 100% 会执行完成.】
    # loop 已经被close()关闭，调用run_until_complete()方法将出现异常
    try:
        print("tij")
        # 若此时调用 loop.run_until_complete(task) ：事件循环会被重新启动，专门用于执行该任务直到完成。
        loop.run_until_complete(task) #由于任务已被取消，执行到这里会触发 CancelledError 异常
        print("2")
    except asyncio.CancelledError as e:
        print("任务已取消.",e,"任务状态:",task.done(),"这里可能做一些清理工作！")
        pass
    # loop.close()方法用于关闭事件循环。一旦调用了close()方法，事件循环将不再接受新的任务，并且会释放所有相关的资源，如文件描述符、线程等。
    # close()方法只能在事件循环停止后调用，如果在事件循环正在运行时调用close()，会引发RuntimeError。
    print("*** loop close ***"*10)
    loop.close() 
# close


print("_"*20)
async def running_task(flag):
    print("任务开始...")
    await asyncio.sleep(2)
    print("任务完成.")
    return flag
loop = asyncio.new_event_loop()
try:
    # 任务执行完成后,后调用 loop.stop() 方法来停止事件循环，
    # 返回结果为 running_task任务返回的结果
    result=loop.run_until_complete(running_task(123))
    print("任务结果:", result)
finally: 
    # 确保事件循环停止后再关闭
    print("Loop is running:", loop.is_running())
    if loop.is_running():
        loop.stop()
    loop.close()
    print("Loop closed")
