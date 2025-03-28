import threading
import asyncio
import select as EventSelect
from multiprocessing import Pipe, Process
from multiprocessing.connection import PipeConnection
from co6co.utils import log
import threading
import time


def recv_with_timeout(conn: PipeConnection, timeout: int):
    if EventSelect.select([conn], [], [], timeout)[0]:
        return conn.recv()
    else:
        raise TimeoutError("Recv timed out")


def worker(self, conn: PipeConnection):
    event: asyncio.Event = self.app.ctx. quit_event
    log.warn("worker start", event, id(event), type(event))
    while True:
        log.warn("worker wait")
        try:
            quit = asyncio.run(asyncio.wait_for(event.wait(), 1))  # 等待事件
            log.warn("worker quit is:", quit)
            if (quit):
                break
            data = recv_with_timeout(conn, 5)   # 接收数据
            log.warn("worker recv data", data)
            conn.send("ok")
        except TimeoutError:
            log.warn("worker timeout")
            continue


def sender(conn: PipeConnection):
    while True:
        conn.send("Hello from sender process")
        # 检查管道中是否有数据可读
        # 当对方关闭连接且管道中没有剩余数据时，poll() 方法会返回 False。
        if conn.poll():  # 检查是否有数据可读
            message = conn.recv()  # 接收数据
            print("Sender received message:", message)
        else:
            time.sleep(0.1)
            print("Sender no message")
            break
        # time.sleep(1)  # 发送间隔
    conn.close()  # 关闭连接


def receiver(conn: PipeConnection):
    while True:
        try:
            # 当发送方关闭连接后，recv() 方法会抛出 EOFError 异常
            message = conn.recv()  # 接收数据
            print("接受到数据:", message)
        except EOFError:
            print("Receiver received EOF")
            break
        except Exception as e:
            print("Receiver received error:", e, type(e))
            break
        finally:
            conn.close()  # 关闭连接


def process_worker():
    """
    检查对方关闭连接
    1. conn.poll()
    2. try: conn.recv() except EOFError:
    """

    parent_conn, child_conn = Pipe()
    p1 = Process(target=sender, args=(child_conn,))

    # 创建接收进程
    p2 = Process(target=receiver, args=(parent_conn,))

    # 启动进程
    p1.start()
    p2.start()

    # 等待进程结束
    p1.join()
    p2.join()


if __name__ == '__main__':
    process_worker()
