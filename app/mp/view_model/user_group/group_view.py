from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sqlalchemy import  func
from sqlalchemy.sql import Select

from sanic import  Request 
from sanic.response import text,raw

from co6co_sanic_ext.utils import JSON_util 
from co6co.utils import log

from view_model.base_view import  AuthMethodView
from model.pos.right import UserGroupPO
from view_model.biz.alarm_ import AlarmFilterItems,AlarmCategoryFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result 

class Groups_View(AuthMethodView): 
    async def get(self,request:Request): 
        """
        用户组 [{id:1,name:站点}]
        """
        try:  
            async with request.ctx.session as session,session.begin():   
                session:AsyncSession=session  
                select=(
                    Select(UserGroupPO.id,UserGroupPO.name)
                )
                executer=await session.execute(select)  
                result = executer.mappings().all()
                result = [dict(a) for a in result]
            result=Result.success(result  )   
        except Exception as e: 
            result=Result.fail(message=f"请求失败：{e}")  
        return JSON_util.response(result)

class Group_View(AuthMethodView): 
    async def get(self,request:Request,pk:int):
        return text("未实现")