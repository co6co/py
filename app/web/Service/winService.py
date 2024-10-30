from co6co_win.services import Winservice
# from app import main
import subprocess
import random
import time
from pathlib import Path
from typing import IO
import os

from co6co.utils import read_stream, log
from co6co.task.thread import Executing
import signal
import psutil


def kill_process_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


class AppWinService(Winservice):
    _svc_name_ = "sysWebService"
    _svc_display_name_ = "python sys webservice"
    _svc_description_ = "webservice"
    # _exe_name_ = "C:\\Users\\Administrator\\envs\\win32\\pythonservice.exe"

    def __init__(self, args):
        super().__init__(args)
        self.process = None
        if not os.path.exists("./logs"):
            os.makedirs("./logs")

    def start(self):
        self.isrunning = True

    def stop(self):
        self.isrunning = False
        # 这些方法只会终止该进程
        # self.process.send_signal(signal.SIGTERM)  # 请求终止进程
        # self.process.kill()

        # 获取进程 ID
        pid = self.process.pid
        # 终止进程树
        kill_process_tree(pid, False)
        self.process.terminate()

    def main(self):
        """
        执行自己的代码
        """
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # 激活虚拟环境
        venv_path = 'C:\\Users\\Administrator\\Envs\\wechat\\Scripts\\activate'
        python_path = 'C:\\Users\\Administrator\\Envs\\wechat\\Scripts\\python.exe'
        with open("D:\\services\\logs\\sys_web_info.log", 'wb') as out_file, open("d:\\services\\logs\\sys_web_error.log", 'wb') as err_file:
            self.process = subprocess.Popen("{}&{} D:\\services\\sys\\app.py".format(venv_path, python_path), env=env, stdout=out_file, stderr=err_file)
            self.process.wait()  # 等待子进程结束

        '''
        while self.isrunning:
            random.seed()
            x = random.randint(1, 1000000)
            Path(f"C:/{x}.txt").touch()
            time.sleep(5)
        Path(f'c:/1.txt').touch()
        '''


if __name__ == '__main__':
    AppWinService.parse_command_line()
