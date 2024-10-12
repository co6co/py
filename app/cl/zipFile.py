import time
import sys
import zipfile
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
import argparse


import datetime
from co6co.task.thread import Executing
from co6co.task.pools import limitThreadPoolExecutor
from co6co.utils.log import progress_bar, warn, __getMessage


class zipEncode():
    pwd_exe_total: int = 0
    start_time: datetime.datetime = None
    pwd_total: int = 0
    findFlag: bool = True
    finshed: bool = False

    def __init__(self) -> None:
        self.pwd_exe_total = 0
        self.start_time = None
        self.pwd_total = 0
        self.findFlag = False
        self.finshed = False
        pass

    def extract(self, file: zipfile.ZipFile, password: str):
        """
        解压文件
        """
        if self.findFlag:
            return
        file.extractall(path='.', pwd=password.encode('utf-8'))

    def result(self, f: Future):
        """
        当密码正确后后面的线程还来不及退出时  f.exception 不出错
        """
        exception = f.exception()
        self. pwd_exe_total = self.pwd_exe_total+1
        if self. pwd_exe_total == self.pwd_total:
            self.finshed = True
            time.sleep(1)
        if self.findFlag:
            return
        if not exception:
            # 如果获取不到异常说明破解成功
            print('find:', f.pwd)
            self.findFlag = True
            return

    async def show(self):
        try:
            while not self.finshed and not self.findFlag:
                end = datetime.datetime.now()
                s = (end-self.start_time).total_seconds()
                if s and self.pwd_total:
                    progress_bar(self.pwd_exe_total/self.pwd_total, "{}个/s".format(str(self.pwd_exe_total // s)))
                time.sleep(1)

        except Exception as e:
            print("Show 错误", e)

    def start(self, filePath):
        # 创建一个标志用于判断密码是否破解成功
        self.flag = True
        # 创建一个线程池
        pool = limitThreadPoolExecutor(max_workers=4, thread_name_prefix="zip_pj")
        nums = [str(i) for i in range(10)]
        # chrs = [chr(i) for i in range(65, 91)]
        # 生成数字+字母的6位数密码
        password_lst = itertools.permutations(nums, 6)
        # password_lst = ['65'+''.join(i) for i in password_lst]
        # 创建文件句柄
        zfile = zipfile.ZipFile(filePath, 'r')
        self.pwd_total = len(password_lst)
        self.start_time = datetime.datetime.now()
        Executing("进度条", self.show)
        print("total:", self.pwd_total)

        for pwd in password_lst:
            if self.findFlag:
                break

            pwd = ''.join(pwd)
            f = pool.submit(self.extract, zfile, pwd)
            f.pwd = pwd
            f.pool = pool
            f.add_done_callback(self.result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成字典")
    group = parser.add_argument_group("获取配置")
    parser.add_argument("-f", "--file",  type=str, help="zipFile")
    args = parser.parse_args()
    if args.file == None:
        parser.print_help()
    else:
        encode = zipEncode()
        encode.start(args.file)
