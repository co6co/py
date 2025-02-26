import threading
from queue import Queue
import time
from co6co.task import ThreadTask
from typing import Generator, Callable, Any


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
    for i in range(2):  # 假设有10个任务
        yield f"Task-{i}"


if __name__ == "__main__":
    task = ThreadTask(process_task, task_generator, oneWorkerEndBck=lambda: print("一个工作线程结束"), taskEndBck=lambda x: print("任务结束", x))
    task.start()
    task.stop()
