from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sanic import  Request 
from sanic.response import text,raw
from sqlalchemy import func

from co6co_sanic_ext.utils import JSON_util,json

from view_model.base_view import  AuthMethodView
from model.pos.biz import bizAlarmPO,bizAlarmAttachPO,bizResourcePO
from view_model.biz.alarm_ import AlarmFilterItems
from co6co_sanic_ext.model.res.result import Page_Result 
from sqlalchemy.sql import Select

class Alarms_View(AuthMethodView):
    """
    告警列表
    """
    async def post(self,request:Request):
        """
        获取列表 
        """ 
        filterItems=AlarmFilterItems()
        filterItems.__dict__.update(request.json)   
        filterItems.__dict__.update(request.json)  
        async with request.ctx.session as session:  
            session:AsyncSession=session  
            select=(
                 Select(bizAlarmPO) 
                .options(joinedload(bizAlarmPO.alarmTypePO)) 
                .filter(and_(*filterItems.filter()))
                .limit(filterItems.limit).offset(filterItems.offset) 
            ) 
            result= await session.execute(select) 
            select=(
                Select(func.count(bizAlarmPO.id) ) 
                .options(joinedload(bizAlarmPO.alarmTypePO)) 
                .filter(and_(*filterItems.filter())) 
            ) 
            total= await session.execute(select) 
            pageList=Page_Result.success(result)
            session.commit()
            #pageList.total=total 
        '''
        opt=DbPagedOperations(session,filterItems) 
        total = await opt.get_count(bizAlarmPO.id)
        select=opt._create_paged_select(filterItems,bizAlarmPO) 
        select=opt.join(select,bizAlarmAttachPO,bizAlarmAttachPO.id==bizAlarmPO.id)
        result = await opt._get_list()   
        pageList=Page_Result.success(result)
        pageList.total=total 
        await opt.commit()
        '''
        return JSON_util.response(pageList)

class Alarm_View(AuthMethodView):
    """
    告警列表
    """
    async def get(self,request:Request,pk:int): 
        return text("未配置")

