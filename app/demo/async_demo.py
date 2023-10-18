# -*- encoding:utf-8 -*-

import asyncio ,time,types
from co6co.utils import log


log.start_mark("异步函数简单示例")
# Future 对象属于可等待对象
async def set_after(fut,delay,value):
    #在协程函数中，可以通过await语法来挂起自身的协程，并等待另一个协程完成直到返回结果
    #await语法只能出现在通过async修饰的函数[协程函数]中
    #await后面的对象需要是一个Awaitable或者实现了相关的协议
    await asyncio.sleep(delay)
    print("输出异步执行结果")
    fut.set_result(value)

def compute_add(x:int,y:int)->int:
    time.sleep(2)
    return x+y

async def main():
    loop=asyncio.get_running_loop() # 当前系统线程中正在运行的事件循环
    future=loop.create_future() #创建个可等待对象
    loop.create_task(set_after(future,2,"..."))

    print(compute_add(2,8)) # 执行其他任务

    print(await future)


asyncio.run(main())

log.end_mark("异步函数简单示例")
#_______________________________________________________________________________________
log.start_mark("基础知识")
def generator():
    yield 1
# 异步生成器函数
async def async_generator():
    yield 1
# 直接调用异步函数不会返回结果，而是返回一个coroutine对象
# coroutine对象 需要通过其他方式来驱动，因此可以使用这个协程对象的send方法给协程发送一个值
async def async_compute_add(x:int,y:int)->int:
    time.sleep(2)
    return x+y

try:
    #生成器/协程在正常返回退出时会抛出一个StopIteration异常，
    # 而的返回值会存放在StopIteration对象的value属性中
    async_compute_add(33,44).send(None)
except StopIteration as e:
    print(e.value) 

print(f"生成器：{type(generator())}，{type(generator()) is types.GeneratorType}")
print(f"异步生成器：{type(async_generator())}，{type(async_generator()) is types.AsyncGeneratorType}")


# async with [异步上下文管理器  ]
# AsyncContextManager: __aenter__ 和 __aexit__ 返回 awaitable类型的值
'''
async def commit(session,data):
    ...
    async with session.transation():
        ...
        await session.update(data)
        ...

#使用锁
aync with lock:
    ...

'''

log.end_mark("基础知识")



