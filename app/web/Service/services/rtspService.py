
from threading import Thread

from co6co.task.eventDispatcher import EventHandler,EventType,Event
from services.ff_service import ffService
import asyncio
from typing import Optional
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol


class RtspService(EventHandler):
    def __init__(self,event:asyncio.BaseEventLoop):
        super().__init__()
        self.ffService = ffService()
        self.event=event
     
    def _stop_stream(self,key:str): 
        try:
            self.ffService.stop(key)
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":True})
        except Exception as e:
            print(f"[ERROR] 停止流失败: {e}")
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":False})
         
    async def _start_get_data(self,key:str,rtsp_url:str):
        #async for data in self.ffService.read_rtsp_stream(rtsp_url,key): 
        async for data in self.ffService.exec_ipconfig( ): 
           
            self.send(self.create_event("RTSP.STREAM.DATA",{"key":key,"data2":rtsp_url})) 

    def _start_thread(self,key:str,rtsp_url:str):  
        data=self.create_event("RTSP.STREAM.DATA",{"key":key,"data":rtsp_url.encode()})
        data=data.to_dict()
        print("发送数据:",data)
        def worker(key, rtsp_url):
            loop =self.event# asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_get_data(key, rtsp_url))
            print(f"[INFO] 流: {key} 结束，关闭事件循环")
            loop.close()
        Thread(target=worker, args=(key, rtsp_url)).start()  
        #asyncio.run(self._start_get_data(key, rtsp_url))

    def handle(self, event: Event) -> Optional[Event]:
        if event.event_type == "RTSP.SERVICE.START":
            try:
                self._start_thread(event.data["key"],event.data["rtsp_url"]) 
            except Exception as e:
                print(f"[ERROR] 启动流失败: {e}")
            print("1111111111",event.data["key"])
        elif event.event_type == "RTSP.SERVICE.STOP":
            self._stop_stream(event.data["key"])
        return None
    @property
    def supported_events(self):
        return ['RTSP.SERVICE.START','RTSP.SERVICE.STOP']

class RtspServiceClient(EventHandler):
    def __init__(self ):
        super().__init__()
        self.k_ws={}
      
    def send_start(self ,ws: WebSocketCommonProtocol,key:str, rtsp_url:str):
        self.k_ws[key]=ws
        print(f"[INFO] 启动流: {key}")
        event = self.create_event("RTSP.SERVICE.START",{"key":key,"rtsp_url":rtsp_url})
        self.send(event)
    def send_stop(self,key:str):
        print(f"[INFO] 停止流: {key}")
        event = self.create_event("RTSP.SERVICE.STOP",{"key":key})
        self.send(event) 
    
    def handle(self, event: Event) -> Optional[Event]:
        key=event.data["key"]
        ws:WebSocketCommonProtocol=self.k_ws[key]
        if event.event_type == "RTSP.STREAM.DATA": 
            data=event.data["data"] 
            ws.send(data)
        elif event.event_type == "RTSP.SERVICE.STOP.RESULT": 
            del self.k_ws[key]
            ws.close()
        return None
    @property 
    def supported_events(self):
        return ['RTSP.STREAM.DATA',"RTSP.SERVICE.STOP.RESULT"]

       
