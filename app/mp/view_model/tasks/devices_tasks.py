import asyncio
from co6co.utils import log,json_util
from sqlalchemy import Select
from model.filters.DeviceFilterItems import posterTaskFilterItems
from utils.cvUtils import screenshot 
from sqlalchemy.ext.asyncio import AsyncSession
from model.pos.biz import bizCameraPO 
from sqlalchemy import or_, and_, Select
from co6co_db_ext.db_session import db_service
from co6co_db_ext.db_operations import DbOperations
import json,os
from typing import List
from co6co.utils.File import File 
from sanic import Blueprint, Sanic
import requests 
from co6co.task.thread import ThreadEvent
from threading import Thread
 
import time
from services.bll import baseBll

class DemoTest( ):
    app:Sanic
    def __init__(self,app:Sanic) -> None:
        self.app=app
        pass
    def checkService(self):
        try:  
            response=requests.get("http://127.0.0.1:8084/v1/api/test",timeout=3)  
            if response.status_code==200:return True
            return False
        except Exception as e:
            print("error:",e)
            return False
    def run(self):
        try:
            app=self.app
            '''
            log.start_mark("tast state")
            print("Name:", app.m.name)
            print("PID:", app.m.pid)
            print("状态：", app.m.state)
            print("workers:", app.m.workers)
            log.end_mark("tast state")
            #app.m.terminate() # 关闭整个应用及其所有的进程  
            '''
            isRuning=self. checkService()
            if  not isRuning: 
                print(">>>> 服务未能提供服务，即将重启 worker...")
                app.m.restart() # 仅重启 worker
            else:
                print("service is runing.") 
            #app.m.name.restart("","") # 重启特点的 worker 
        except Exception as e:
            log.err("檢測任務失敗,即将重启 worker...")
            app.m.restart() # 仅重启 worker


class stream:
    name:str
    url:str
    def __init__(self,name,url) -> None:
        self.name=name
        self.url=url
        pass
 

async def update_device_poster_task(app:Sanic|Blueprint): 
    
    filter=posterTaskFilterItems()
    service:db_service=app.ctx.service 
    while True:
        session:AsyncSession=None
        try: 
            await asyncio.sleep(300)	# 设定任务休眠时间 
            #t=ThreadEvent()     
            bll=DemoTest(app ) 
            Thread(target=bll.run() ).start()
            log.warn("this is 1..")
            #log.warn("tasking")
            #time.sleep(60)
            #log.warn("tasked")
            continue
            session:AsyncSession=service.async_session_factory()
            sanrc=await session.execute(filter.count_select)
            count=sanrc.scalar() 
            log.succ(f"获取设备视频poster：{count}")
            if count>0 and filter.pageIndex<=filter.getMaxPageIndex(count): 
                await queryData(session,filter)
                filter.pageIndex+=1
            else:filter.pageIndex=1
            '''
            sanrc:bizCameraPO=await session.get_one(bizCameraPO,ident= 2)
            data=[ stream("高清","http://wx.co6co.top:452/flv/vlive/2.flv").__dict__]
            sanrc.streams=json.dumps(data,ensure_ascii=False)
            ''' 
        except Exception as e:
            if session!=None:
                await session.rollback() 
            log.err(f"执行定时任务失败：{e}")
        finally:
            if session!=None:
                await session.close()
        

async def queryData(session:AsyncSession,filter:posterTaskFilterItems): 
   
    query=await session.execute(filter.list_select)
    rows=query.scalars().all()  
    for row in rows:
        try:
            row:bizCameraPO=row 
            data=json.loads(row.streams) 
            data:List[stream] =[stream(**a) for a in data] 
            d=data.pop()
            print(d,type(d),d.name,d.url)
            if row.poster !=None and os.path.exists(row.poster):os.remove(row.poster) 
            log.succ(f"截图{d.url}")
            row.poster= await screenshot(d.url ) 
            log.succ(row.poster)
            await session.commit()
        except Exception as e:
            if session!=None: await session.rollback() 
            log.err(f"执行定时任务失败：{e}")

