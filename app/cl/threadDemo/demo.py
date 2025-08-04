from co6co.task.thread import ThreadEvent

import asyncio


async def task():
    print("开始任务..")
    await asyncio.sleep(15)
    print("任务完成")
    return 15

t = ThreadEvent("线程003", lambda: print("线程003结束"))
result = t.runTask(task)
print(result)
