from co6co.task.thread import ThreadEvent
from co6co.task.pools import limitThreadPoolExecutor
from co6co.utils import network
import asyncio


async def task(ip):
    print("开始任务..") 
    pingResult = network .ping_host(ip)
    print("任务完成",pingResult)
    return pingResult

t = ThreadEvent("线程003", lambda: print("线程003结束"))

#for i in range(1,255):
#    ip=f"192.168.1.{i}"
#    result = t.runTask(task,ip)
#    print(f"ping {ip}，{result   }") 
#print(result)

def bck(f):
    print(f"ping {f.ip}，{f.result()}")

with limitThreadPoolExecutor(4) as executor:
    for i in range(1,255):
        ip=f"192.168.1.{i}"
        future=executor.submit(network .ping_host,ip)
        future.ip=ip
        future.add_done_callback(bck)

        
