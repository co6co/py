# pip install pywin32
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
from co6co.task.thread import Executing


class PySvc(win32serviceutil.ServiceFramework):
    _svc_name_ = 'PySvc'
    _svc_display_name_ = 'Python Service'
    _svc_description_ = 'This is a Python service.'

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)

        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        try:
            servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                                  servicemanager.PYS_SERVICE_STARTED,
                                  (self._svc_name_, ''))
        except Exception as e:
            print("Eeee")
        # Executing("服务线程", self.main)
        self.main()

    def main(self):
        # 这里是你的服务主逻辑
        while True:
            if win32event.WaitForSingleObject(self.hWaitStop, 2000) == win32event.WAIT_OBJECT_0:
                break
            else:
                # 执行你的任务
                print("123")
                pass

    def biz():
        '''
        业务逻辑
        '''
        pass


if __name__ == '__main__':
    print("python winServiceApp.py install --安装服务")
    print("python winServiceApp.py start --开始服务")
    print("python winServiceApp.py stop --开始服务")
    print("python winServiceApp.py remove -- --删除服务")
    print(sys.argv)

    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(PySvc)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(PySvc)
