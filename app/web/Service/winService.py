from co6co_win.services import Winservice
import subprocess
import os
import win32serviceutil


class AppWinService(Winservice):
    _svc_name_ = "sysWebService"
    _svc_display_name_ = "python sys webservice"
    _svc_description_ = "webservice"
    # _exe_name_ = str(PYTHON_PATH.joinpath("pythonservice.exe"))
    # _exe_args_ = '-u -E "' + os.path.abspath(__file__) + '"'
    isrunning: bool = None

    def __init__(self, args):
        super().__init__(args)
        self.process = None
        self.isrunning = False
        self.logFolder = "E:\Tools\\www\\system\\logs"
        self.virtualEnvPath = "C:\\Users\\Administrator\\Envs\\wechat"
        self.appPath = "E:\Tools\\www\\system\\service\\app.py"
        if not os.path.exists(self.logFolder):
            os.makedirs(self.logFolder)

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
        self.kill_process_tree(pid, False)
        self.process.terminate()

    def main(self):
        """
        执行自己的代码
        """
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # 激活虚拟环境
        venv_path = os.path.join(self.virtualEnvPath, 'Scripts\\activate')
        python_path = os.path.join(self.virtualEnvPath, 'Scripts\\python.exe')
        env['PATH'] = os.path.join(self.virtualEnvPath, 'Scripts\\python.exe;') + os.environ['PATH']
        infoFile = os.path.join(self.logFolder, "sys_web_info.log")
        errorFile = os.path.join(self.logFolder, "sys_web_error.log")
        with open(infoFile, 'wb') as out_file, open(errorFile, 'wb') as err_file:
            param = {"env": env, "stdout": out_file, "stderr": err_file}
            self.process = subprocess.Popen("{} {}".format(python_path, self.appPath), **param)
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
    # 虚拟环境中
    win32serviceutil.HandleCommandLine(AppWinService)
    # 系统中的 python 环境
    # AppWinService.parse_command_line()
