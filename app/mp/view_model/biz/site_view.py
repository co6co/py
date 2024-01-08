from sqlalchemy.ext.asyncio import AsyncSession,AsyncSessionTransaction
import asyncio
from co6co_db_ext .db_operations import DbOperations
from co6co_db_ext.db_utils import db_tools
from sanic import  Request 
from sanic.response import text,raw,empty,file_stream
from co6co_sanic_ext.utils import JSON_util
import json
from model.filters.SiteFilterItems import SiteDiveceFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from view_model import get_upload_path
from view_model.base_view import BaseMethodView, AuthMethodView
from model.pos.biz import bizResourcePO,bizSitePo
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

        try: 
            log.err(f"session ..1，{id( request.ctx.session)}") 
            async with request.ctx.session as session,session.begin():  
                log.err(f"session ..2")
                session:AsyncSession=session  
                total=await session.execute(filterItems.count_select)
                log.err(f"session ..3")
                total=total.scalar()  
                execute= await session.execute(filterItems.list_select)
                result=execute.unique().scalars().all()  
            data=[] 
            for a in result:
                d={ }
                a:bizSitePo=a 
                for c in a.__table__.columns:
                    print(type(c),c)
                    
                    print("label:", c.label.__name__,"\t_label:",c._label)
                    print("key:", c.key,"\t_key_label:",c._key_label)
                    print("anon_label:", c.anon_label,"\t_key_label:",c._anon_label)
                    print("key:", c._key_label,"\t_anon_key_label:",c._anon_key_label)
                    print("key:", c._anon_name_label,"\t_key_label:",c._anon_tq_label)
                    log.warn(c.anon_key_label)

                d.update(a.to_dict()) 
                devices=[]
                a.boxPO
                if a.boxPO:d.update({"box":a.boxPO.to_dict()})
                for pa in a.camerasPO: 
                    devices.append(pa.to_dict())
                d.update({"devices":devices})
                data.append(d)  
            pageList=Page_Result.success(data ,total=total)   
        except Exception as e:
            log.err(f"session ... e:{e}")
            pageList=Page_Result.fail(message=f"请求失败：{e}")
            raise
        except asyncio.CancelledError:
            # 处理任务被取消后的逻辑
            print("Task was cancelled.")
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

