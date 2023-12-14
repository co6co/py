from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sqlalchemy import func
from sqlalchemy.sql import Select

from sanic import  Request 
from sanic.response import text,raw

from co6co_sanic_ext.utils import JSON_util,json
from co6co.utils import log

from view_model.base_view import  AuthMethodView
from model.pos.biz import bizAlarmPO,bizAlarmAttachPO,bizResourcePO,BasePO
from view_model.biz.alarm_ import AlarmFilterItems
from co6co_sanic_ext.model.res.result import Page_Result 


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
        async with request.ctx.session as session:  
            session:AsyncSession=session 
            opt=DbOperations(session)  
            select=(
                 Select(bizAlarmPO) 
                .options(joinedload(bizAlarmPO.alarmTypePO)) 
                .options(joinedload(bizAlarmPO.alarmAttachPO)) 
                .filter(and_(*filterItems.filter()))
                .limit(filterItems.limit).offset(filterItems.offset)
            )  
            result= await opt._get_list(select,True) 
            select=(
                Select( func.count( )).select_from(
                    Select(bizAlarmPO.id) 
                    .options(joinedload(bizAlarmPO.alarmTypePO))
                    .filter(and_(*filterItems.filter()))
                )
            ) 
            total= await opt._get_scalar(select)  
            pageList=Page_Result.success(result,total=total)  
            await session.commit() 
        return JSON_util.response(pageList)

class Alarm_View(AuthMethodView):
    """
    告警列表
    """
    async def get(self,request:Request,pk:int): 
        return text("未配置")

