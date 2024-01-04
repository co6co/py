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
 
import time

class stream:
    name:str
    url:str
    def __init__(self,name,url) -> None:
        self.name=name
        self.url=url
        pass
    
    
async def update_device_poster_task(app): 
    filter=posterTaskFilterItems()
    service:db_service=app.ctx.service

    while True:
        session:AsyncSession=None
        try: 
            await asyncio.sleep(800)	# 设定任务休眠时间 
            #log.warn("tasking")
            #time.sleep(60)
            #log.warn("tasked")
            #continue
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

