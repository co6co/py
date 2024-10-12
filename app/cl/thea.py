from co6co.task.thread import ThreadEvent
import asyncio
import time
from co6co.task.thread import TaskManage


async def bck():
    print("123456")
    loop = asyncio.get_running_loop()
    print(loop, id(loop))
    return True


if __name__ == "__main__":
    # loop = asyncio.get_running_loop()
    # print(loop, id(loop))
    t = TaskManage("theard1")

    def stop(r):
        print("停止")
        t.stop()

    t.runTask(bck, stop)
    while True:
        time.sleep(1)
        if not t.runing:
            t.close()
            break
    print("等待关闭")
