from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,joinedload
from sqlalchemy import and_,func 
from sqlalchemy.sql import Select 

from sanic import  Request 
from sanic.response import text,raw

from co6co_sanic_ext.utils import JSON_util
import json
from co6co.utils import log

from view_model.base_view import  AuthMethodView,BaseMethodView
from model.pos.biz import bizAlarmPO,bizAlarmAttachPO,bizResourcePO,BasePO
from view_model.biz.alarm_ import AlarmFilterItems,AlarmCategoryFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result 
from co6co_db_ext .db_utils import db_tools
from sqlalchemy.sql.elements import ColumnElement

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

class Alarm_View_Base(BaseMethodView): #AuthMethodView
    async def _get_one(self,request:Request,*filters:ColumnElement[bool]): 
        """
        查询单条告警
        """
        session:AsyncSession=request.ctx.session 
        filterItems=AlarmFilterItems()
        async with session,session.begin():     
            execute= await session.execute(filterItems.list_select.filter(and_(*filters)))
            result=execute.mappings().fetchone()
            pageList=Result.success(db_tools.one2Dict(result) )   
        return JSON_util.response(pageList)

class Alarms_View(Alarm_View_Base): 
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
            opt=DbOperations(session)    
            execute= await session.execute(filterItems.list_select)
            result=execute.scalars().all()  
            result= await opt._get_tuple(filterItems.list_select)
            #result=[dict(a)  for a in  result]    
            pageList=Page_Result.success(result,total=total)  
            await session.commit() 
        return JSON_util.response(pageList)

class Alarm_View(Alarm_View_Base):
    """
    告警页
    """
    async def get(self,request:Request,pk:int): 
        return text("为实现")
    
    async def post(self,request:Request,pk:str): 
        """
        查询单条告警
        """
        return await self._get_one(request,bizAlarmPO.id==pk) 
    
class Alarm_uuid_View(Alarm_View_Base):
    async def post(self,request:Request,uuid:str): 
        """
        查询单条告警
        """  
        return await self._get_one(request,bizAlarmPO.uuid==uuid) 

    
    
