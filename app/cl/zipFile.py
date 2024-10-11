import time
import sys
import zipfile
import itertools
from concurrent.futures import ThreadPoolExecutor
import argparse

import queue
from concurrent.futures import ThreadPoolExecutor
import datetime
from co6co.task.thread import Executing


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """
    将无xian队列改成有界队列，这样就不会出现内存爆满的问题
    """

    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        # self._work_queue = queue.Queue(self._max_workers * 2)  # 设置队列大小


def progress_bar(i: float, title: str = ""):
    i = int(i*100)
    print("\r", end="")
    print("{}: {}%: ".format(title, i), "▋" * (i // 2), end="")
    sys.stdout.flush()


g_flage = 0
g_start = None
g_tatol = 0
flag = True


def extract(file, password: str):
    global flag
    if not flag:
        return
    # print(password, type(password))
    file.extractall(path='.', pwd=password.encode('utf-8'))  # .encode('utf-8')


def result(f):
    """
    当密码正确后后面的线程还来不及退出时  f.exception 不出错
    """
    global g_flage, flag
    exception = f.exception()
    g_flage = g_flage+1
    if not flag:
        return
    if not exception:
        # 如果获取不到异常说明破解成功
        print('密码为：', f.pwd)
        flag = False


async def show():
    try:
        global flag, g_start, g_tatol
        while flag:
            end = datetime.datetime.now()
            s = (end-g_start).total_seconds()
            if s and g_tatol:
                progress_bar(g_flage/g_tatol, "{}个/s".format(str(g_flage // s)))
            time.sleep(1)

    except Exception as e:
        print("Show 错误", e)


def main(filePath):
    # 创建一个标志用于判断密码是否破解成功
    global flag, g_tatol, g_start
    flag = True
    # 创建一个线程池
    pool = BoundedThreadPoolExecutor(8)
    nums = [str(i) for i in range(10)]
    # chrs = [chr(i) for i in range(65, 91)]
    # 生成数字+字母的6位数密码
    password_lst = itertools.permutations(nums, 6)
    password_lst = ['65'+''.join(i) for i in password_lst]
    # password_lst = ["65358691"]
    # 创建文件句柄
    zfile = zipfile.ZipFile(filePath, 'r')
    g_tatol = len(password_lst)
    g_start = datetime.datetime.now()
    Executing("进度条", show)
    for pwd in password_lst:
        if not flag:
            break

        pwd = ''.join(pwd)
        f = pool.submit(extract, zfile, pwd)
        f.pwd = pwd
        f.pool = pool
        f.add_done_callback(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成字典")
    group = parser.add_argument_group("获取配置")
    parser.add_argument("-f", "--file",  type=str, help="zipFile")
    args = parser.parse_args()
    if args.file == None:
        parser.print_help()
    else:
        main(args.file)
