'''
创建windows 服务
----------------------------------------- 
 
1. 第一如下变量
   _svc_name_ = "服务名"
   _svc_display_name_ = "显示名"
   _svc_description_ = "描述"
2. 重写方法:
    def start(self)  
    def stop(self)   
    def main(self)  主要业务逻辑 
3. 主方法中调用 cls.parse_command_line  
'''
import socket
import win32serviceutil
import servicemanager
import win32event
import win32service
import psutil
from abc import abstractmethod, ABC


class Winservice(win32serviceutil.ServiceFramework, ABC):
    '''Base class to create winservice in Python'''
    _svc_name_ = 'pythonService'
    _svc_display_name_ = 'Python Service'
    _svc_description_ = 'Python Service Description'

    @classmethod
    def parse_command_line(cls):
        '''
        类方法
        解析命令行
        '''
        win32serviceutil.HandleCommandLine(cls)

    @staticmethod
    def kill_process_tree(pid, including_parent=True):
        """
        杀进程树
        :param pid: 进程id
        :param including_parent: 是否包括父进程
        """
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        psutil.wait_procs(children, timeout=5)
        if including_parent:
            parent.kill()
            parent.wait(5)

    def __init__(self, args):
        '''
        构造函数
        '''
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        '''
        Win服务调用停止入口
        '''
        self.stop()
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        '''
        Win服务调用开始入口
        '''
        self.start()
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    @abstractmethod
    def start(self):
        '''
        Override to add logic before the start
        eg. running condition
        '''
        pass

    @abstractmethod
    def stop(self):
        '''
        Override to add logic before the stop
        eg. invalidating running condition
        '''
        pass

    @abstractmethod
    def main(self):
        '''
        Main class to be ovverridden to add logic
        '''
        pass
