from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw,empty,file_stream
from co6co_sanic_ext.utils import JSON_util,json

from view_model import get_upload_path
from view_model.base_view import  AuthMethodView
from model.pos.right import bizResourcePO
import os 

class Resources_View(AuthMethodView):
    """
    资源视图
    """
    async def get(self,request:Request):
        return text("未实现")
    
class Resource_View(AuthMethodView):
    """
    资源视图
    """
    async def get(self,request:Request,uuid:str):
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session)
            while(True): 
                url= await operation.get_one(bizResourcePO.url,bizResourcePO.uid==uuid) 
                if url==None:break
                else:
                    upload=get_upload_path(request.app.config)
                    fullPath=os.path.join(upload,url) 
                    if not os.path.exists(fullPath):break
                    return await file_stream(fullPath ) 
        return empty(status=404)