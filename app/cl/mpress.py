from multiprocessing import Process, Queue
# 主子线程同讯
## 1. Queue
def worker(q):
    q.put("Hello from worker process")

if __name__ == "__main__":
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    print(q.get())  # 主进程中接收信息
    p.join()

## 2. PIPE
from multiprocessing import Process, Pipe

def worker(conn):
    conn.send("Hello from worker process")
    conn.close()

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p = Process(target=worker, args=(child_conn,))
    p.start()
    print(parent_conn.recv())  # 主进程中接收信息
    p.join()

## 3. 共享内存
from multiprocessing import Process, Value, Array

def worker(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == "__main__":
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=worker, args=(num, arr))
    p.start()
    p.join() 

## 4.Manager

from multiprocessing import Process, Manager

def worker(d, l):
    d[1] = '1'
    d['2'] = 2
    l.reverse()

if __name__ == "__main__":
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=worker, args=(d, l))
        p.start()
        p.join()

        print(d)
        print(l)