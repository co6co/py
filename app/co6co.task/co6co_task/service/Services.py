from co6co_sanic_ext import sanics
from sanic import Sanic
import asyncio
from multiprocessing.connection import PipeConnection

from co6co.task.eventDispatcher import EventDispatcher
class TasksMgr(sanics.IWorker):
    def __init__(self, app: Sanic, envent: asyncio.Event, conn: PipeConnection):
        self.app = app
        self.envent = envent
        self.conn = conn
        self.eventDispatcher = EventDispatcher()
        self.register_handler()
    def start(self):
       self.eventDispatcher.start()
    def stop(self):
       self.eventDispatcher.stop()

