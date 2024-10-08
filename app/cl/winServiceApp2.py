
import time
import win32event
import win32service
import win32serviceutil
from abc import ABC, abstractmethod
import socket


class PythonService(win32serviceutil.ServiceFramework, ABC):
    @classmethod
    def parse_command_line(cls):
        ''' Parse the command line '''
        win32serviceutil.HandleCommandLine(cls)
    # Override the method in the subclass to do something just before the service is stopped.

    @abstractmethod
    def stop(self):
        pass
    # Override the method in the subclass to do something at the service initialization.

    @abstractmethod
    def start(self):
        pass
    # Override the method in the subclass to perform actual service task.

    @abstractmethod
    def main(self):
        pass

    def __init__(self, args):
        ''' Class constructor'''
        win32serviceutil.ServiceFramework.__init__(self, args)
        # Create an event which we will use to wait on.
        # The "service stop" request will set this event.
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        '''Called when the service is asked to stop'''
        # We may need to do something just before the service is stopped.
        self.stop()
        # Before we do anything, tell the SCM we are starting the stop process.
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        # And set my event.
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        '''Called when the service is asked to start. The method handles the service functionality.'''
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        # We may do something at the service initialization.
        self.start()
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        # Starts a worker loop waiting either for work to do or a notification to stop, pause, etc.
        self.main()


class MointorImageService(PythonService):
    # Define the class variables
    _svc_name_ = "MointorImageOnLandingPageService"
    _svc_display_name_ = "MSSQLTips Mointor Images Service"
    _svc_description_ = "Mointor images on the landing page of the MSSQLTips.com."
    _exe_name_ = "C:\\Users\\Administrator\\envs\\win\\pythonservice.exe"
    # Override the method to set the running condition

    def start(self):
        self.isRunning = True
    # Override the method to invalidate the running condition
    # When the service is requested to be stopped.

    def stop(self):
        self.isRunning = False
    # Override the method to perform the service function

    def main(self):
        while self.isRunning:
            # app.check_web_page_images()
            time.sleep(5)


# Use this condition to determine the execution context.
if __name__ == '__main__':
    # Handle the command line when run as a script
    MointorImageService.parse_command_line()
