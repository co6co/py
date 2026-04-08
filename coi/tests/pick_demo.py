
# 1. 检查数据是否可序列化
import pickle
import os
from co6co.task.eventDispatcher import Event
from multiprocessing import Pipe
from multiprocessing.connection import PipeConnection
import multiprocessing 
#在 Linux 和 macOS 上，通常不需要调用 freeze_support()，因为这两个系统使用 fork来创建子进程，不会重新导入主模块
#下面代码 即使在非 Windows 平台调用也无害
multiprocessing.freeze_support()  # 在 Windows 上使用多进程时需要
# 1. 启动模式
## 1.1  只有 Spawn 会把子进程 __name__ 改成 __mp_main__
## Fork、Forkserver完全不修改 __name__，子进程依然是 __main__
## Fork 复制主进程内存空间（写时复制），子进程直接继承主进程的所有状态、模块、变量、线程，不重新导入主模块
##    子进程是 “克隆体”，不是重新运行脚本，if __name__ == '__main__' 块在主进程已经执行过，子进程不会再执行，无无限递归风险

def check_name():
    # 手动指定启动模式（Windows 只能选 spawn）
    # multiprocessing.set_start_method('fork')  # Linux/macOS可用
    # multiprocessing.set_start_method('forkserver')  # Linux/macOS可用
    print(f"PID={os.getpid()} | 启动模式={multiprocessing.get_start_method()} | __name__={__name__}")
data=Event.create("abcdd", "test_dispatcher", "test_task")
data=data.to_dict()
try:
    serialized = pickle.dumps(data )
    print(data ,"可序列化：",serialized)
except Exception as e:
    print(f"数据不可序列化: {e}")

# 2. 估算大小
data_size = len(pickle.dumps(data))
if data_size > 100 * 1024 * 1024:  # 100MB
    print("数据太大，考虑使用共享内存")

# 3. 使用分块传输
def send_large_data_safely(conn:PipeConnection, data, chunk_size=1024 * 1024):
    """安全地发送大数据""" 
    serialized = pickle.dumps(data)
    
    # 发送大小信息
    # conn.send(len(serialized))
    
    # 分块发送
    for i in range(0, len(serialized), chunk_size):
        conn.send(serialized[i:i+chunk_size])
    
    # 确认
    conn.send(b"DONE")

def receive_large_data_chunked(conn:PipeConnection,result:list):
    """分块接收大文件"""
    # 接收元数据
    #metadata = conn.recv()
    #total_size = metadata["total_size"]
    #chunk_size = metadata["chunk_size"]
    
    # 接收数据
    chunks = []
    received_size = 0
    
    while True:
        chunk = conn.recv()
        if chunk is None:  # 结束标记
            break
        chunks.append(chunk)
        received_size += len(chunk)
    
    # 合并数据
    #return b"".join(chunks)
    result.append(b"".join(chunks))



if __name__ == '__main__':  #__mp_main__
    workers :list[multiprocessing.Process]= [] 
    parent_conn, child_conn =  Pipe()
    worker = multiprocessing.Process(
        target=send_large_data_safely,
        args=( parent_conn, data)
    )
    workers.append(worker)
    result=[]
    worker2 = multiprocessing.Process(
        target=receive_large_data_chunked,
        args=( child_conn, result)
    )
    workers.append(worker2)
    # 启动所有worker
    for worker in workers:
        worker.start()

    # 等待
    for worker in workers:
        worker.join(timeout=10)
    print(result)
    for item in result:
        print(item)
        deserialized = pickle.loads(item)
        print("反序列化后的数据:", deserialized)