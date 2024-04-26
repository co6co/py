
from sanic.views import HTTPMethodView,stream # 基于类的视图
from sanic.response import text,redirect
from sanic import  Request  
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Page_Result, Result  
from co6co_db_ext.db_operations import DbOperations
from sqlalchemy.ext.asyncio import AsyncSession

from co6co .utils import log
from datetime import datetime

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils  import db_tools,  DbCallable,QueryOneCallable, QueryListCallable,QueryPagedByFilterCallable
from co6co.utils.tool_util import list_to_tree 

from .base_view import AuthMethodView
from ..model.pos.right import UserGroupPO
from ..model.filters.user_group_filter import user_group_filter


class user_groups_tree_view(AuthMethodView):
    async def get(self, request:Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """ 
        select=(
            Select(UserGroupPO.id,UserGroupPO.name,UserGroupPO.code,UserGroupPO.parentId)  
            .order_by(UserGroupPO.parentId.asc())
        ) 
        return await self.query_tree(request,select,  pid_field='parentId',id_field="id",isPO=False) 
    
    async def post(self, request:Request):
        """
        树形 table数据
        tree 形状 table
        """ 
        param=user_group_filter()
        param.__dict__.update(request.json)   
        return await self.query_tree(request,param.list_select,rootValue=0, pid_field='parentId',id_field="id") 


class user_groups_view(AuthMethodView):
    async def get(self, request:Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """ 
        select=(
            Select(UserGroupPO.id,UserGroupPO.name,UserGroupPO.code,UserGroupPO.parentId)  
            .order_by(UserGroupPO.parentId.asc())
        ) 
        return await self.query_list(request,select,  isPO=False) 
    
    async def post(self, request:Request):
        """
        树形 table数据
        tree 形状 table
        """ 
        param=user_group_filter()
        param.__dict__.update(request.json)   
        return await self.query_list(request,param.list_select ) 
    
    async def put(self,request:Request):  
        """
        增加
        """  
        po=UserGroupPO()
        userId=self.getUserId(request)
        async def before( po:UserGroupPO, session:AsyncSession,request):   
            exist=await db_tools.exist(session,  UserGroupPO.code.__eq__(po.code),column=UserGroupPO.id) 
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
        return await self.add(request,po,userId,before) 

    def patch(self, request:Request):
        return text("I am patch method")

class user_group_view(AuthMethodView):
    async def put(self,request:Request,pk:int):  
        """
        编辑
        """  
        po=UserGroupPO()
        async def before(oldPo:UserGroupPO, po:UserGroupPO, session:AsyncSession,request):  
            exist=await db_tools.exist(session, UserGroupPO.id!=oldPo.id,UserGroupPO.code.__eq__(po.code),column=UserGroupPO.id)
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
            if po.parentId==oldPo.id:return JSON_util.response(Result.fail(message=f"'父节点选择错误！") )
            oldPo.code=po.code
            oldPo.name=po.name
            oldPo.parentId=po.parentId

        return await self.edit(request,pk,po,UserGroupPO,self.getUserId(request),before) 
    async def delete(self,request:Request,pk:int):  
        """
        删除
        """   
        return await self.remove(request,pk,UserGroupPO ) 
     


     
    