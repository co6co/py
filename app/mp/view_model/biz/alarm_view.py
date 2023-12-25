from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sqlalchemy import  func
from sqlalchemy.sql import Select

from sanic import  Request 
from sanic.response import text,raw

from co6co_sanic_ext.utils import JSON_util
import json
from co6co.utils import log

from view_model.base_view import  AuthMethodView
from model.pos.biz import bizAlarmPO,bizAlarmAttachPO,bizResourcePO,BasePO
from view_model.biz.alarm_ import AlarmFilterItems,AlarmCategoryFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result 

class Alarm_category_View(AuthMethodView): 
    """
    告警类型
    """
    async def get(self,request:Request): 
        """
        告警类型
        """ 
        param = AlarmCategoryFilterItems() 
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer = await session.execute(param.create_List_select()) 
            result = executer.mappings().all()
            result = [dict(a) for a in result]
            pageList = Result.success(result )
            await session.commit()
        return JSON_util.response(pageList)  

class Alarms_View(AuthMethodView):
    """
    告警
    """
    async def post(self,request:Request):
        """
        获取列表 
        """ 
        filterItems=AlarmFilterItems()
        filterItems.__dict__.update(request.json)
        #return JSON_util.response(Page_Result.fail())
        async with request.ctx.session as session:  
            session:AsyncSession=session  
            total=await session.execute(filterItems.count_select)
            total=total.scalar()
            #result=await session.execute(filterItems.list_select)
            # total= result.scalars().fetchall()  
            opt=DbOperations(session)  
            result=await opt._get_list(filterItems.list_select,True)  
            pageList=Page_Result.success(result,total=total)  
            await session.commit() 
        return JSON_util.response(pageList)

class Alarm_View(AuthMethodView):
    """
    告警列表
    """
    async def get(self,request:Request,pk:int): 
        return text("未配置")

