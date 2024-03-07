from model.enum import hwx_alarm_type
from services.bll import baseBll
from view_model.base_view import BaseMethodView,Request 
from sanic.response import text,raw
from co6co .utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from model.pos.biz import  bizAlarmPO,bizBoxPO
import time  
from datetime import datetime
from co6co.task.thread import ThreadEvent
from typing  import Tuple

class TestBll(baseBll):
    async def get(self): 
        async with  self.session,self.session.begin():
            select=(Select(bizAlarmPO).filter(bizAlarmPO.id>0))
            exec=await self.session.execute(select)
            po:Tuple[bizAlarmPO]=exec.fetchone()
            for a in po: 
                a.createTime=datetime.now()
                time.sleep(5) 
        return text("执行完毕2。")


    
class TestView(BaseMethodView):
    def get(self,request:Request ): 
        return text(f"请求成功，你可以试试其他的:{request.args}") 
    
    async def post(self,request:Request):
        """
        session:sleep:10
        """
        session:AsyncSession=request.ctx.session
        async with  session,session.begin():
            select=(Select(bizAlarmPO).filter(bizAlarmPO.id>0))
            exec=await session.execute(select)
            map=exec.mappings().all()
            for m in map:
                print(m) 
        return text("执行完毕。")
    
    async def put (self,request:Request):
        """
        session:sleep:10
        """ 
        session:AsyncSession=request.ctx.session
        async with  session,session.begin():
            select=(Select(bizAlarmPO).filter(bizAlarmPO.id>0))
            exec=await session.execute(select)
            po:bizAlarmPO=exec.fetchone()
            for a in po:
                print(a) 
                print(type(a),a.createTime)
                a.createTime=datetime.now()
                time.sleep(15) 
        return text("执行完毕2。")
    async def delete (self,request:Request):
        """
        session:sleep:10
        """ 
        t=ThreadEvent()     
        bll=TestBll(request.app,t.loop)
        await bll.get()
        return text("执行完毕2。")
    
from view_model.biz.upload_view import syncCheckEntity,createResourceUUID,saveResourceToDb,alarm_success

class TestsView(BaseMethodView):
    def get(self,request:Request ):
        po=bizAlarmPO()
        po.alarmType=hwx_alarm_type.alarm1.key
        po.alarmTime=datetime.now()
        alarm_success(request,po)
        return text(f"请求成功，你可以试试其他的:{request.args}") 
        