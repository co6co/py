import multiprocessing
import time
from multiprocessing.connection import Listener, Client
import threading

def worker(address, worker_id):
    # 每个工作进程作为客户端连接服务器
    with Client(address, authkey=b'secret') as conn:
        while True:
            msg = conn.recv()
            if msg == "exit":
                print(f"Worker {worker_id} exiting.")
                break
            print(f"Worker {worker_id} received: {msg}")

if __name__ == "__main__":
    address = ('localhost', 6000)
    
    # 启动监听器（在独立线程中）
    def listener_func():
        listener = Listener(address, authkey=b'secret')
        connections = {}
        
        for i in range(3):
            conn = listener.accept()
            connections[i] = conn
            print(f"Worker {i} connected")
        
        # 向指定连接发送消息
        connections[0].send("Hello worker 0")
        connections[1].send("Hello worker 1")
        
        time.sleep(1)
        
        for i in range(3):
            connections[i].send("exit")
            connections[i].close()
        
        listener.close()
    
    listener_thread = threading.Thread(target=listener_func)
    listener_thread.start()
    
    # 启动工作进程
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=worker, args=(address, i))
        p.start()
        processes.append(p)
        time.sleep(0.1)  # 确保连接顺序
    
    for p in processes:
        p.join()
    
    listener_thread.join()