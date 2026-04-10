# 主进程与多个子进程 通讯
# 多个通道方案
import multiprocessing
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
import time

def worker(conn:Connection, worker_id):
    while True:
        msg = conn.recv()
        if msg == "exit":
            print(f"Worker {worker_id} exiting.")
            conn.close()
            break
        pid=os.getpid()
        print(f"{pid}Worker {worker_id} received: {msg}")
        #time.sleep(1)

if __name__ == "__main__":
    num_workers = 3
    connections:dict[int,Connection] = {}
    processes = {}
    
    # 为每个子进程创建管道
    for i in range(num_workers):
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=worker, args=(child_conn, i))
        p.start()
        connections[i] = parent_conn
        processes[i] = p
    
    # 向指定进程发送消息
    connections[0].send("Hello Worker 0!")  # 只发给worker 0
    connections[1].send("Hello Worker 1!")  # 只发给worker 1
    connections[2].send("Hello Worker 2!")  # 只发给worker 1
    
    time.sleep(2)
    
    # 发送退出信号
    for i in range(num_workers):
        connections[i].send("exit")
        processes[i].join()