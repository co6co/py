# 有错
import multiprocessing
import time
from multiprocessing.managers import BaseManager

# 创建队列管理器
class QueueManager(BaseManager):
    pass

# 注册队列类型
QueueManager.register('get_queue')

def create_queue_manager(port=5000, authkey=b'secret'):
    """创建队列管理器服务"""
    queue_dict = {}
    
    def get_queue(queue_id):
        if queue_id not in queue_dict:
            queue_dict[queue_id] = multiprocessing.Queue()
        return queue_dict[queue_id]
    
    QueueManager.register('get_queue', get_queue)
    manager = QueueManager(address=('localhost', port), authkey=authkey)
    return manager

def worker(worker_id, port=5000, authkey=b'secret'):
    """工作进程，从管理器获取队列"""
    QueueManager.register('get_queue')
    manager = QueueManager(address=('localhost', port), authkey=authkey)
    manager.connect()
    
    # 获取自己的队列
    my_queue = manager.get_queue(worker_id)
    
    while True:
        if not my_queue.empty():
            msg = my_queue.get()
            if msg == "exit":
                print(f"Worker {worker_id} exiting.")
                break
            print(f"Worker {worker_id} received: {msg}")
        time.sleep(0.1)

if __name__ == "__main__":
    num_workers = 3
    port = 5000
    authkey = b'secret'
    
    # 启动管理器服务器
    manager_server = create_queue_manager(port, authkey)
    server = manager_server.get_server()
    
    # 在后台启动服务器
    from multiprocessing import Process
    server_process = Process(target=server.serve_forever)
    server_process.daemon = True
    server_process.start()
    
    time.sleep(1)  # 等待服务器启动
    
    # 创建工作进程
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(i, port, authkey))
        p.start()
        processes.append(p)
    
    # 父进程也连接到管理器
    QueueManager.register('get_queue')
    manager = QueueManager(address=('localhost', port), authkey=authkey)
    manager.connect()
    
    # 向指定进程发送消息
    for i in range(num_workers):
        q = manager.get_queue(i)
        q.put(f"Message for worker {i}")
    
    time.sleep(1)
    
    # 发送退出信号
    for i in range(num_workers):
        q = manager.get_queue(i)
        q.put("exit")
    
    for p in processes:
        p.join()
    
    server.shutdown()