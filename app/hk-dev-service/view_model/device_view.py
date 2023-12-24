
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sanic import  Request 
from sanic.response import text,raw

from sqlalchemy import func,and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session

from co6co_sanic_ext.utils import JSON_util 
from co6co.utils import log
from model.enum.task import Task_Statue, Task_Type
from model.enum.config import sys_config_Enum
 
from view_model import AuthMethodView

from co6co_db_ext.db_operations import DbOperations
from co6co_sanic_ext.model.res.result import Page_Result ,Result
from sqlalchemy.sql import Select
 
from model.filters.deviceFilter import DeviceFilterItems, CategoryFilterItems, LightFilterItems
from model.enum import device_type
from model.pos.device import devicePo,TasksPO,sysConfigPO
import datetime
from co6co_db_ext.db_session import db_service
from services.hik_service import DemoTest

class Device_Category_View(AuthMethodView):  
    async def get(self,request:Request):
        """
        获取设备列表
        """
        param=CategoryFilterItems() 
        session:scoped_session=request.ctx.session  
       
        result= session.execute(param.list_select)   
        result=result.mappings().all() 
        result=[dict(a)  for a in result] 
        pageList=Result.success(result  )   
        return JSON_util.response(pageList)
    
class Device_View(AuthMethodView):  
    async def post(self,request:Request):
        """
        获取设备列表
        """
        param=DeviceFilterItems()
        param.__dict__.update(request.json) 
        
        session:scoped_session=request.ctx.session  
       
        result= session.execute(param.list_select)   
        result=result.mappings().all() 
        result=[dict(a)  for a in result]
        
        executer= session.execute(param.count_select)  
        pageList=Page_Result.success(result,total= executer.scalar() )   
        return JSON_util.response(pageList)
    
    async def run_setting_task(self, app,param:LightFilterItems,total:int,userId:int,taskName:str):
        log.start_mark("开始设置任务")
        allPageIndex=param.getMaxPageIndex(total)
        service:db_service=app.ctx.service 
        session:scoped_session=service.session
        try:
            tpo= TasksPO () 
            tpo.createUser=userId
            tpo.name=taskName
            tpo.type=Task_Type.down_task.val
            tpo.status=Task_Statue.created.val
            session.add(tpo)
            session.commit()  
            upo=session.execute(sysConfigPO).filters(sysConfigPO.code==sys_config_Enum.hk_username.key) 
            ppo:sysConfigPO=session.get(sysConfigPO,sysConfigPO.code==sys_config_Enum.hk_password.key)
            if upo==None or ppo==None: raise Exception(f"{sys_config_Enum.hk_username.key} 或者 {sys_config_Enum.hk_password.key}未配置")
            for i in range(1,allPageIndex+1): 
                    param.pageIndex=i
                    execute=session.execute(param.list_select)
                    result=execute.scalars().all()
                    for po in result:
                        po:devicePo=po
                        log.info(po.name)
                        po.taskState=1
                        
            
            tpo.status=Task_Statue.finshed.val
        except Exception as e: 
            tpo.status=Task_Statue.error.val
            log.err(f"执行任务失败：{e}")
        finally:
            session.commit()
            session.close_all()  
        log.start_mark("结束设置任务")



    async def patch(self,request:Request):
        """
        创建设置补光灯任务
        """ 
        param=LightFilterItems()
        param.__dict__.update(request.json) 
        current_user_id=request.ctx.current_user["id"]
        session:scoped_session=request.ctx.session  
        executer= session.execute(param.count_select)  
        total=executer.scalar()

            
        name=f"设置补光灯任务_{datetime.datetime.now().strftime('%H%M%S%f')}"
        request.app.add_task(self. run_setting_task(request.app,param,total,current_user_id,name),name=name)
        return JSON_util.response(Result.success(data= f"记录条数：{total},任务名称为:{name}"))
        
    