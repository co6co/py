
from threading import Thread
from co6co_task.service.taskMgr import TaskMgr
from co6co.task.eventDispatcher import EventHandler,EventType,Event
from services.ff_service import ffService
import asyncio
from typing import Optional
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol


class RtspService(EventHandler):
    def __init__(self):
        super().__init__()
        self.ffService = ffService()
     
    def _stop_stream(self,key:str): 
        try:
            self.ffService.stop(key)
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":True})
        except Exception as e:
            print(f"[ERROR] 停止流失败: {e}")
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":False})
         
    async def _start_get_data(self,key:str,rtsp_url:str):
        async for data in self.ffService.read_rtsp_stream(rtsp_url,key): 
            self.send(self.create_event("RTSP.STREAM.DATA",{"key":key,"data":data})) 

    def _start_thread(self,key:str,rtsp_url:str):
        def worker(key, rtsp_url):
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_get_data(key, rtsp_url))
        Thread(target=worker, args=(key, rtsp_url)).start() 

    def handle(self, event: Event) -> Optional[Event]:
        if event.event_type == "RTSP.SERVICE.START":
            self._start_thread(event.data["key"],event.data["rtsp_url"]) 
        elif event.event_type == "RTSP.SERVICE.STOP":
            self._stop_stream(event.data["key"])
        return None
    def supported_events(self):
        return ['RTSP.SERVICE.START','RTSP.SERVICE.STOP']

class RtspServiceClient(EventHandler):
    def __init__(self ):
        super().__init__()
        self.k_ws={}
      
    def send_start(self ,ws: WebSocketCommonProtocol,key:str, rtsp_url:str):
        self.k_ws[key]=ws
        event = self.create_event("RTSP.SERVICE.START",{"key":key,"rtsp_url":rtsp_url})
        self.send(event)
    def send_stop(self,key:str):
        event = self.create_event("RTSP.SERVICE.STOP",{"key":key})
        self.send(event)
    
    def handle(self, event: Event) -> Optional[Event]:
        if event.event_type == "RTSP.STREAM.DATA":
            key=event.data["key"]
            data=event.data["data"]
            ws=self.k_ws[key]
            ws.send(data)
        elif event.event_type == "RTSP.SERVICE.STOP.RESULT":
            key=event.data["key"]
            ws=self.k_ws[key]
            del self.k_ws[key]
            ws.close()
        return None
    def supported_events(self):
        return ['RTSP.STREAM.DATA',"RTSP.SERVICE.STOP.RESULT"]

       
