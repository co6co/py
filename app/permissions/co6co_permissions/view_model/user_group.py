
from sanic.response import text
from sanic import  Request  
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import  Result   
from sqlalchemy.ext.asyncio import AsyncSession 

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils  import db_tools
 

from .base_view import AuthMethodView
from ..model.pos.right import UserGroupPO
from ..model.filters.user_group_filter import user_group_filter


class user_groups_tree_view(AuthMethodView):
    routePath="/tree"
    
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
        if len( param.filter())>0:
            return await self.query_page(request,param)
        return await self.query_tree(request,param.create_List_select(),rootValue=0, pid_field='parentId',id_field="id") 


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
        return await self.add(request,po,userId= userId,beforeFun= before) 

    def patch(self, request:Request):
        return text("I am patch method")

class user_group_view(AuthMethodView):
    routePath="/<pk:int>"
    
    async def put(self,request:Request,pk:int):  
        """
        编辑
        """   
        async def before(oldPo:UserGroupPO, po:UserGroupPO, session:AsyncSession,request):  
            exist=await db_tools.exist(session, UserGroupPO.id!=oldPo.id,UserGroupPO.code.__eq__(po.code),column=UserGroupPO.id)
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
            if po.parentId==oldPo.id:return JSON_util.response(Result.fail(message=f"'父节点选择错误！") ) 

        return await self.edit(request,pk,UserGroupPO,userId= self.getUserId(request),fun=before) 
    async def delete(self,request:Request,pk:int):  
        """
        删除
        """   
        async def before(po:UserGroupPO,session:AsyncSession):
            count=await db_tools.count(session  ,UserGroupPO.parentId==po.id ,column=UserGroupPO.id)
            if count>0:return JSON_util.response(Result.fail(message=f"该'{po.name}'节点下有‘{count}’节点，不能删除！"))
        return await self.remove(request,pk,UserGroupPO,beforeFun=before ) 
     


     
    