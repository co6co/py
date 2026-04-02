import asyncio
import random
from asyncio import Semaphore
from co6co.task.utils import Timer,clock

### demo1 等待单个异步操作
async def fetch_data():
    print("开始获取数据")
    await asyncio.sleep(2)  # 模拟耗时操作
    print("数据获取完成")
    return {"data": "example"}

async def main():
    result = await fetch_data()  # 等待 fetch_data 完成
    print(f"结果: {result}")


## demo2 等待多个异步任务
async def task(taskIndex,taskTime):
    print(f"任务{taskIndex}开始...{taskTime}")
    await asyncio.sleep(taskTime)
    print(f"任务{taskIndex}完成.")
    
    return {"taskIndex":taskIndex,"taskTime":taskTime}
 

async def main2():
    # 创建任务
    # 同时启动三个任务，总耗时约2秒（最长的任务时间）
    timer=Timer()
    timer.start()
    results = await asyncio.gather(
        task(1,1),
        task(2,2),
        task(3,0.5),
    )
    timer.stop()
    print(f"所有结果: {results}，耗时: {timer.elapsed} 秒") 

# demo3  等待多个异步任务 asyncio.wait()- 更灵活的控制
async def main3():
    # 创建任务
    
    tasks = [asyncio.create_task(task(i,random.uniform(0.5,2)))  for i in range(5)]
    
    timer=Timer()
    timer.start()
    # 等待所有任务完成
    # asyncio.wait()期望接收的是 任务（Task）对象，而不是原始的协程（coroutine）对象。
    done, pending = await asyncio.wait(tasks, timeout=1.5)
    timer.stop()
    for t in done:
        print(f"结果: {t.result()}")
    print("*"*20)
    print(f"asyncio.wait()   期望接收的是 任务(Task)对象,而不是原始的协程(coroutine)对象。")
    print(f"已完成: {len(done)} 个")
    print(f"未完成: {len(pending)} 个")

    print(f"耗时: {timer.elapsed} 秒")
    print("*"*20)
    # 处理已完成的任务
    
# demo 4 等待特定条件的任务
async def main4():
    try:
        # 最多等待2秒
        print(f"asyncio.wait_for()  期望接收任务(Task)对象  或者 原始的协程(coroutine)对象。")
        result = await asyncio.wait_for(asyncio.create_task(task('我执行需要5s',1.5)), timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        print("操作超时！")

## 5 使用 asyncio.as_completed()按完成顺序处理
async def process_tasks():
    tasks = [
        asyncio.create_task(task('按完成顺序处理_任务_1',1)),
        asyncio.create_task(task('按完成顺序处理_任务_2',2)),
        asyncio.create_task(task('按完成顺序处理_任务_3',0.5)),
        task('按完成顺序处理_任务_6',1),
        task('按完成顺序处理_任务_7',1),
    ]
    
    # 按照完成顺序处理
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"收到结果: {result}")

# 6 并发限制
async def limited_task(sem, task_id,taskTime):
    async with sem:  # 控制并发数
        return await task(task_id,taskTime) 

async def main6():
    sem = Semaphore(3)  # 最多3个并发
    tasks = [limited_task(sem, i,random.uniform(0.5,2)) for i in range(10)]
    results =await asyncio.gather(*tasks)
    print(results)


while True:
    index=input("q退出,输入序号: 1. 等待单个异步操作,2. 等待多个异步操作完成,3. asyncio.wait() 更灵活的控制,4.任务超时,5.按完成顺序处理任务,6.并发限制\n:")
    if index=="q":
        break 
    if index=="1":
        asyncio.run(main())

    elif index=="2":
        asyncio.run(main2())
        
    elif index=="3":
        asyncio.run(main3())
    elif index=="4":
        asyncio.run(main4())
    elif index=="5":
        asyncio.run(process_tasks())
    elif index=="6":
        asyncio.run(main6())
    else:
        print("输入错误")
    print("")
        
 