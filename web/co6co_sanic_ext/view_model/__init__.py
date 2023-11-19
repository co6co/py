from functools import wraps 
from sanic.views import HTTPMethodView # 基于类的视图
from sanic import  Request
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations,DbOperations,InstrumentedAttribute
from co6co_sanic_ext.model.res.result import Page_Result
from co6co_sanic_ext.utils import  JSON_util
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
            await session.commit()
            return JSON_util.response(pageList)
"""
class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 
"""