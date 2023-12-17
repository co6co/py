
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sanic import  Request 
from sanic.response import text,raw

from sqlalchemy import func,and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session

from co6co_sanic_ext.utils import JSON_util 
from co6co.utils import log
 
from view_model import AuthMethodView

from co6co_db_ext.db_operations import DbOperations
from co6co_sanic_ext.model.res.result import Page_Result 
from sqlalchemy.sql import Select

from model.pos import device 
from model.filters.deviceFilter import DeviceFilterItems
from model.enum import device_type



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

    async def patch(self,request:Request):
        """
        创建任务
        """
        log.start_mark("创建下载任务")
        param=ProcesFilterItems()
        param.__dict__.update(request.json) 
        current_user_id=request.ctx.current_user["id"]
        async with request.ctx.session as session: 
            if param.groupIds and len ( param.groupIds)>0 : 
                opt=DbOperations(session)
                result= await queryBoatSerials(param.groupIds,opt) 
                for id in result:
                    if id not in param.boatSerials:
                        param.boatSerials.append(id)  
            opt=DbPagedOperations(session,param)  
            total = await opt.get_count(ProcessPO.id) 
            name=f"下载任务_{datetime.datetime.now().strftime('%H%M%S%f')}"
            request.app.add_task(downloadTask( request.app,param ,total,current_user_id,name),name=name) 
        await session.commit()
        log.end_mark("创建下载任务")
        return JSON_util.response(Result.success(data= f"记录条数：{total},任务名称为:{name}"))
        
    