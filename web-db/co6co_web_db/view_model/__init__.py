from functools import wraps 
from sanic.views import HTTPMethodView # 基于类的视图
from sanic import  Request
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations,DbOperations,InstrumentedAttribute
from co6co_sanic_ext.model.res.result import Page_Result
from co6co_sanic_ext.utils import  JSON_util
from sqlalchemy.ext.asyncio import AsyncSession
from typing import TypeVar
from co6co_sanic_ext.model.res.result import Result
#from api.auth import authorized

class BaseMethodView(HTTPMethodView):  
    async def _get_list(self,request:Request,filterItems:absFilterItems,field:InstrumentedAttribute="*" ):
        """
        列表
        """ 
        filterItems.__dict__.update(request.json)  
        async with request.ctx.session as session:  
            opt=DbPagedOperations(session,filterItems) 
            total = await opt.get_count(field) 
            result = await opt.get_paged()   
            pageList=Page_Result.success(result)
            pageList.total=total 
            await opt.commit()
            return JSON_util.response(pageList)
    async def _del_po(self,request:Request,poType:TypeVar,pk:int ): 
        """
        删除数据库对象
        """ 
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session)
            po= await operation.get_one_by_pk(poType,pk) 
            if po==None:return JSON_util.response(Result.fail(message=f"未该'{pk}'对应得数据!"))  
            await operation.delete(po) 
            await operation.commit()    
            return JSON_util.response(Result.success())
"""
class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 
"""