from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw,empty,file_stream
from co6co_sanic_ext.utils import JSON_util
import json
from model.filters.SiteFilterItems import SiteDiveceFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from view_model import get_upload_path
from view_model.base_view import BaseMethodView, AuthMethodView
from model.pos.biz import bizResourcePO
import os 
from co6co.utils import log
from sqlalchemy.engine.row import RowMapping

class Sites_View(AuthMethodView):
    """
    安全员站点
    """
    async def post(self,request:Request):
        """
        列表 
        """ 
        filterItems=SiteDiveceFilterItems()
        filterItems.__dict__.update(request.json)
        #return JSON_util.response(Page_Result.fail())
        async with request.ctx.session as session:  
            session:AsyncSession=session  
            total=await session.execute(filterItems.count_select)
            total=total.scalar()  
            execute= await session.execute(filterItems.list_select)
            result=execute.unique().mappings().fetchone()
            #result= await opt._get_tuple(filterItems.list_select) 
            log.warn(type(result))
            
            log.warn(result.values())
            pageList=Page_Result.success(result.values(),total=total)  
            await session.commit() 
        return JSON_util.response(pageList)
    
class Site_View(BaseMethodView):
    """
    资源视图
    """
    async def get(self,request:Request,uid:str):
        """
        获取资源内容
        """ 
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session) 
            while(True):  
                url= await operation.get_one(bizResourcePO.url,bizResourcePO.uid==uid) 
                if url==None:break 
                else:
                    upload=get_upload_path(request.app.config)
                    fullPath=os.path.join(upload,url[1:]) 
                    if not os.path.exists(fullPath):break 
                    await operation.commit()
                    #file(s,mime_type="image/jpeg")  
                    return await file_stream(fullPath ) 
        return empty(status=404)

