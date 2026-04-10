import multiprocessing
from multiprocessing import Queue
import time,os
from dataclasses import dataclass
from typing import Any
from co6co.utils import log

@dataclass
class TargetedMessage:
    target_id: int
    data: Any

def worker(queue:Queue, worker_id):
    while True:
        pid=os.getpid()
        log.succ(f"{pid} 等待消息...")
        msg = queue.get()
        
        log.succ(f"{pid} 收到消息: {len(msg.data)}")
        if isinstance(msg.data, bytes):
            msg.data = msg.data.decode("utf-8") 
        if isinstance(msg, TargetedMessage) and msg.target_id == worker_id:
            if msg.data == "exit":
                log.succ(f"{pid}Worker {worker_id} exiting.")
                break
            if isinstance(msg.data, bytes):
                msg.data = msg.data.decode("utf-8")
            log.succ(f"{pid}:Worker {worker_id} received: {len(msg.data)}")
        else:
            log.warn(f"{pid} {worker_id}!={msg.target_id} 不是发给自己的消息，放回队列")
            queue.put(msg)
        # 如果不是发给自己的消息，放回队列（注意：实际中需考虑竞争条件）
        # 更好的做法是每个进程有自己的队列，见方案三

if __name__ == "__main__":
    num_workers = 3
    queue = multiprocessing.Queue()
    
    processes = []
    
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    
    # 向指定进程发送消息
    queue.put(TargetedMessage(target_id=0, data="Task for worker 0"))
    queue.put(TargetedMessage(target_id=1, data="Task for worker 1"))
    queue.put(TargetedMessage(target_id=2, data="Task for worker 2"))
    queue.put(TargetedMessage(target_id=2, data=("Task for worker 2你好"*40960).encode()))
    
    time.sleep(10)
    while not queue.empty():
        time.sleep(1)
    
    # 发送退出信号
    for i in range(num_workers):
        queue.put(TargetedMessage(target_id=i, data="exit"))
    
    for p in processes:
        p.join()