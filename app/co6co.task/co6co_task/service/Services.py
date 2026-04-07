from co6co_sanic_ext import sanics
from sanic import Sanic
import asyncio
from multiprocessing.connection import PipeConnection
from co6co.task.eventDispatcher import EventDispatcher,EventDispatcherProcess
from .handler.task_handler import TaskManager, ExistHandler,StartHandler,ModifyHandler,RemoveHandler,GetNextRunTimeHandler,UnknownHandler
class Service(sanics.IWorker):
    """服务类,接受客户端的任务并处理"""
    def __init__(self, app: Sanic, envent: asyncio.Event, conn: PipeConnection):
        self.app = app
        self.envent = envent
        self.conn = conn
        
        self.event_process = EventDispatcherProcess(conn, "service")

        self.task_manager = TaskManager(self.app)  
        self.event_process.append_handler( ExistHandler(self.app)) 
        self.event_process.append_handler( StartHandler(self.app)) 
        self.event_process.append_handler( ModifyHandler(self.app)) 
        self.event_process.append_handler( RemoveHandler(self.app)) 
        self.event_process.append_handler( GetNextRunTimeHandler(self.app)) 
        self.event_process.append_handler( UnknownHandler(self.app))
    def start(self):
       self.event_process.start()
       self.task_manager.start()
    def stop(self):
       self.event_process.stop()
       self.task_manager.stop()

