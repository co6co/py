import multiprocessing
import time
from multiprocessing.managers import BaseManager

from co6co.utils import log
def worker(queue:multiprocessing.Queue, worker_id):
    log.succ(type(dict)) 
    while True:
        if not queue.empty():
            msg = queue.get()
            if msg == "exit":
                print(f"Worker {worker_id} exiting.")
                break
            print(f"Worker {worker_id} received: {msg}")
        time.sleep(0.1)  # 避免忙等待

if __name__ == "__main__":
    num_workers = 3
    manager = multiprocessing.Manager()   

    # 创建共享字典，值为队列
    queues:list[multiprocessing.Queue] = []
    processes:list[multiprocessing.Process] = []
    
    for i in range(num_workers):
        q = multiprocessing.Queue()
        queues.append(q)  
        p = multiprocessing.Process(target=worker, args=(q, i))
        p.start()
        processes.append(p)
    
    # 向指定进程发送消息
    queues[0].put("Message for worker 0")
    queues[1].put("Message for worker 1")
    queues[2].put("Message for worker 2")
    
    time.sleep(1)
    
    # 发送退出信号
    for i in range(num_workers):
        queues[i].put("exit")
    
    for p in processes:
        p.join()