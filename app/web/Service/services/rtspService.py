
from threading import Thread

from co6co.task.eventDispatcher import EventHandler,EventType,Event
from services.ff_service import ffService
import asyncio
from typing import Optional
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
from co6co.task.thread import TaskManage
from co6co.utils import log 
import base64
class RtspService(EventHandler):
    def __init__(self ):
        super().__init__()
        self.ffService = ffService() 
        self.taskMgr=TaskManage(threadName="rtspStreamTask")
        
     
    def _stop_stream(self,key:str): 
        try:
            self.ffService.stop(key)
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":True})
        except Exception as e:
            print(f"[ERROR] 停止流失败: {e}")
            return self.create_event("RTSP.SERVICE.STOP.RESULT",{"key":key,"success":False})
         
    async def _start_get_data(self,key:str,rtsp_url:str): 
        async for data in self.ffService.read_rtsp_stream(rtsp_url,key): 
        #async for data in self.ffService.exec_ipconfig( ): 
            data="你好123456".encode('utf-8')
            print("ffServiceData,",key,len(data))
            # 发送数据到客户端 

            base64_data = base64.b64encode(data) 
            ev=self.create_event("RTSP.STREAM.DATA",{"key":key,"data": base64_data}) 
            #ev = self.create_event("RTSP.STREAM.DATA",{"key":key,"rtsp_url":rtsp_url})
            #print("发送数据:",ev.to_dict()) 
            try:
                self.send(ev) 
            except Exception as e:
                print(f"[ERROR] 发送数据失败: {e}")

    def _start_stream_task(self,key:str,rtsp_url:str):   
        self.taskMgr.runTask(self._start_get_data,lambda x:print(f"[INFO] 流任务: {key} 结束，执行结果：{x.result}"),key, rtsp_url)
       

    def handle(self, event: Event) -> Optional[Event]:
        if event.event_type == "RTSP.SERVICE.START":
            key=event.data["key"]
            url=event.data["rtsp_url"]
            try:
                self._start_stream_task(key,url) 
            except Exception as e:
                print(f"[ERROR] 启动{key}{url}流失败: {e}") 
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
        #log.warn(f"{key}处理事件: {event.to_dict()}")
        ws:WebSocketCommonProtocol=self.k_ws[key]
        if event.event_type == "RTSP.STREAM.DATA":  
            base64_data= event.data.get('data')
            data = base64.b64decode(base64_data) 
            log.info(f"{key}处理数据: {type( data)}{len(data)}") 
            #ws.send(data)
        elif event.event_type == "RTSP.SERVICE.STOP.RESULT": 
            del self.k_ws[key]
            ws.close()
        return None
    @property 
    def supported_events(self):
        return ['RTSP.STREAM.DATA',"RTSP.SERVICE.STOP.RESULT"]

       
