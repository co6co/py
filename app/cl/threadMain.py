import threading
from queue import Queue
import time


def process_task(task):
    """
    模拟任务函数
    """
    print(f"正在处理任务: {task}")
    import time
    time.sleep(2)  # 模拟耗时操作
    print(f"任务完成: {task}")


def task_generator():
    """
    # 生成器函数，用于生成任务
    生成一个字符串
    """
    for i in range(10):  # 假设有10个任务
        yield f"Task-{i}"


def main(max_threads=3):
    """
    主逻辑
    """
    # 创建一个队列，用于存储任务
    task_queue = Queue()

    def worker():
        """
        # 定义线程函数，从队列中取出任务并处理
        """
        while True:
            print("线程函数执行中...")
            # 会阻塞
            task = task_queue.get()  # 从队列中获取任务
            print("线程函数执行中2...")
            if task is None:  # 如果任务为None，表示终止信号
                break
            process_task(task)
            task_queue.task_done()  # 标记任务已完成
        print("线程函数执行完毕")

    # 创建指定数量的线程
    threads = []
    for _ in range(max_threads):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)
    # 使用生成器生成任务，并将其放入队列
    generator = task_generator()
    for task in generator:
        # 等待队列中有空闲位置（即有线程完成任务）
        while task_queue.qsize() >= max_threads:
            time.sleep(0.1)  # 短暂等待，避免高CPU占用
        task_queue.put(task)

    # 等待所有任务完成
    task_queue.join()

    # 发送终止信号给所有线程
    for _ in range(max_threads):
        task_queue.put(None)

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    print("所有任务已完成")


if __name__ == "__main__":
    main(max_threads=3)
