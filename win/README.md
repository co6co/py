# 基础组件

# 发布历史

```
0.0.1 初始版
0.0.2  优化
```

# 安装

```
pip install co6co_win

from pathlib import Path

class WinserviceDemo(Winservice):
    _svc_name_ = "pythonService"
    _svc_display_name_ = "pythonService"
    _svc_description_ = "pythonService"
    # _exe_name_ = "C:\\Users\\Administrator\\envs\\win32\\pythonservice.exe"

    def start(self):
        self.isrunning = True

    def stop(self):
        self.isrunning = False

    def main(self):
        """
        执行自己的代码
        """
        i = 0
        while self.isrunning:
            random.seed()
            x = random.randint(1, 1000000)
            Path(f"C:/{x}.txt").touch()
            time.sleep(5)
        Path(f'c:/1.txt').touch()

# 调用方法
if __name__ == '__main__':
    WinserviceDemo.parse_command_line()
```


# 注意事项 (`pywin32.pth` 目录不用`相对路径`使用`绝对路径`)
a. 使用虚拟环境时,运行服务默认使用默认的python环境，需要修改 pywin32中的相关变量
```
//C:\Users\Administrator\Envs\win32\Lib\site-packages\pywin32.pth
win32
win32\lib
Pythonwin
//-->修改为当前虚拟环境目录-->
C:\Users\Administrator\Envs\win32\Lib\site-packages\win32
C:\Users\Administrator\Envs\win32\Lib\site-packages\win32\lib
C:\Users\Administrator\Envs\win32\Lib\site-packages\Pythonwin
```
