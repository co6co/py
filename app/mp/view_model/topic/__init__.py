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
from model.pos.biz import bizMqttTopicPO
from view_model.biz.alarm_ import AlarmFilterItems,AlarmCategoryFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result 
from co6co_db_ext.db_utils import db_tools

class Topic_View(AuthMethodView):
    """
    安全员站点
    """ 
    async def get(self,request:Request,category:str,code:str):
        """
        获取 
        [{key:1,key2:..}]
        """
        try:   
            session:AsyncSession=request.ctx.session 
            async with session,session.begin():
                select=(
                    Select(bizMqttTopicPO.topic,bizMqttTopicPO.mqttServerId).filter(bizMqttTopicPO.category==category,bizMqttTopicPO.code==code)
                )
                executer=await session.execute(select)  
                result = executer.mappings().fetchone()
                result = dict(result)  
            result=Result.success(result  )   
        except Exception as e: 
            raise
            result=Result.fail(message=f"请求失败：{e}")  
        return JSON_util.response(result)