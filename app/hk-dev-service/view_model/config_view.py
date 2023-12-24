
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sanic import  Request 
from sanic.response import text,raw

from sqlalchemy import func,and_ 
from sqlalchemy.orm import scoped_session

from co6co_sanic_ext.utils import JSON_util 
from co6co.utils import log 
 
from view_model import AuthMethodView
 
from co6co_sanic_ext.model.res.result import Page_Result ,Result
from sqlalchemy.sql import Select
 
from model.filters.configFilter import ConfigFilterItems
from model.enum import device_type
from model.pos.device import sysConfigPO
from datetime import datetime
from co6co_db_ext.db_session import db_service
from services.hik_service import DemoTest


class Configs_View(AuthMethodView):  
    async def post(self,request:Request):
        """
        获取系统配置
        """
        param=ConfigFilterItems()
        param.__dict__.update(request.json)  
        session:scoped_session=request.ctx.session  
       
        result= session.execute(param.list_select)   
        result=result.mappings().all() 
        result=[dict(a)  for a in result]
        
        executer= session.execute(param.count_select)  
        pageList=Page_Result.success(result,total= executer.scalar() )   
        return JSON_util.response(pageList)
    async def put(self, request:Request):
        """
        增加
        """
        po =sysConfigPO()
        po.__dict__.update(request.json)   
        current_user_id=self.getUserId(request)
        session:scoped_session=request.ctx.session   
        po.id=None  
        po.createUser=current_user_id 
        session.add(po) 
        session.commit()
        return JSON_util.response(Result.success()) 
    
class Config_View(AuthMethodView):   
    
    async def put(self, request:Request,pk:int):
        """
        更新
        """
        po =sysConfigPO()
        po.__dict__.update(request.json)   
        current_user_id=self.getUserId(request)
        session:scoped_session=request.ctx.session   
     
        old_po:sysConfigPO= session.get_one(sysConfigPO,pk)
        if old_po==None: return JSON_util.response(Result.fail(message=f"未找{pk},对应的对象!"))  
        old_po.name=po.name
        old_po.code=po.code
        old_po.value=po.value 
        old_po.updateUser=current_user_id
        old_po.updateTime=datetime.now()
        session.commit()
        return JSON_util.response(Result.success()) 

    async def delete(self, request:Request,pk:int):
        """
        删除
        """ 
        session:scoped_session=request.ctx.session    
        old_po:sysConfigPO= session.get_one(sysConfigPO,pk)
        if old_po==None: return JSON_util.response(Result.fail(message=f"未找{pk},对应的对象!"))  
        session.delete(old_po)
        session.commit()
        return JSON_util.response(Result.success()) 
    
    