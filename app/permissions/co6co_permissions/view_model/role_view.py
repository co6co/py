
from sanic.response import text
from sanic import  Request  
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import  Result   
from sqlalchemy.ext.asyncio import AsyncSession 

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils  import db_tools
 

from .base_view import AuthMethodView
from ..model.pos.right import RolePO,MenuRolePO,UserRolePO,UserGroupRolePO
from ..model.filters.role_filter import role_filter
 
class roles_view(AuthMethodView):
    async def get(self, request:Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """ 
        select=(
            Select(RolePO.id,RolePO.name,RolePO.code )  
            .order_by(RolePO.order.asc())
        ) 
        return await self.query_list(request,select,  isPO=False) 
    
    async def post(self, request:Request):
        """
        table数据 
        """ 
        param=role_filter()
        param.__dict__.update(request.json)   
        return await self.query_page(request,param ) 
    
    async def put(self,request:Request):  
        """
        增加
        """  
        po=RolePO()
        userId=self.getUserId(request)
        async def before( po:RolePO, session:AsyncSession,request):   
            exist=await db_tools.exist(session,  RolePO.code.__eq__(po.code),column=RolePO.id) 
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
        return await self.add(request,po,userId= userId,beforeFun= before) 

    def patch(self, request:Request):
        return text("I am patch method")

class role_view(AuthMethodView):
    routePath="/<pk:int>"
    
    async def put(self,request:Request,pk:int):  
        """
        编辑
        """   
        async def before(oldPo:RolePO, po:RolePO, session:AsyncSession,request):  
            exist=await db_tools.exist(session, RolePO.id!=oldPo.id,RolePO.code.__eq__(po.code),column=RolePO.id)
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") ) 

        return await self.edit(request,pk,RolePO,userId= self.getUserId(request),fun=before) 
    async def delete(self,request:Request,pk:int):  
        """
        删除
        """   
        async def before(po:RolePO,session:AsyncSession):
            count=await db_tools.count(session  ,MenuRolePO.roleId==po.id ,column=RolePO.id)
            if count>0:return JSON_util.response(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个菜单，不能删除！"))
            count=await db_tools.count(session  ,UserGroupRolePO.roleId==po.id ,column=RolePO.id)
            if count>0:return JSON_util.response(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个用户组，不能删除！"))
            count=await db_tools.count(session  ,UserRolePO.roleId==po.id ,column=RolePO.id)
            if count>0:return JSON_util.response(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个用户，不能删除！"))
           
        return await self.remove(request,pk,RolePO,beforeFun=before ) 
     


     
    