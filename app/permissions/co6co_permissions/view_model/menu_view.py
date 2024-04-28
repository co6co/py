
from sanic.response import text
from sanic import  Request  
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import  Result   
from sqlalchemy.ext.asyncio import AsyncSession 

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils  import db_tools 
from .base_view import AuthMethodView
from ..model.pos.right import menuPO
from ..model.filters.menu_filter import menu_filter
from ..model.enum import menu_type


class menu_category_view(AuthMethodView):
    """
    菜单类别
    """
    async def post(self, request:Request):
        states=menu_type.to_dict_list() 
        return JSON_util.response(Result.success( data=states)) 

class menu_state_view(AuthMethodView):
    """
    菜单状态
    """
    async def post(self, request:Request):
        return {}

class menu_tree_view(AuthMethodView):
    async def get(self, request:Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """ 
        select=(
            Select(menuPO.id,menuPO.name,menuPO.code,menuPO.parentId)  
            .order_by(menuPO.parentId.asc())
        ) 
        return await self.query_tree(request,select,pid_field='parentId',id_field="id",isPO=False) 
    
    async def post(self, request:Request):
        """
        树形 table数据
        tree 形状 table
        """ 
        param=menu_filter()
        param.__dict__.update(request.json)   
        if len( param.filter())>0:
            return await self.query_list(request,param.list_select) 
        return await self.query_tree(request,param.create_List_select(),rootValue=0, pid_field='parentId',id_field="id") 


class menus_view(AuthMethodView):
    async def get(self, request:Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """ 
        select=(
            Select(menuPO.id,menuPO.name,menuPO.code,menuPO.parentId)  
            .order_by(menuPO.parentId.asc())
        ) 
        return await self.query_list(request,select,  isPO=False) 
    
    async def post(self, request:Request):
        """
        树形 table数据
        tree 形状 table
        """ 
        param=menu_filter()
        param.__dict__.update(request.json)

        return await self.query_list(request,param.list_select ) 
    
    async def put(self,request:Request):  
        """
        增加
        """  
        po=menuPO()
        userId=self.getUserId(request)
        async def before( po:menuPO, session:AsyncSession,request):   
            exist=await db_tools.exist(session,  menuPO.code.__eq__(po.code),column=menuPO.id) 
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
            if type(po.methods)==list: po.methods=",".join(po.methods)
        return await self.add(request,po,userId,before) 

    def patch(self, request:Request):
        return text("I am patch method")

class menu_view(AuthMethodView):
    async def put(self,request:Request,pk:int):  
        """
        编辑
        """
        po=menuPO()
        async def before(oldPo:menuPO, po:menuPO, session:AsyncSession,request):  
            exist=await db_tools.exist(session, menuPO.id!=oldPo.id,menuPO.code.__eq__(po.code),column=menuPO.id)
            if exist:return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！") )
            if po.parentId==oldPo.id:return JSON_util.response(Result.fail(message=f"'父节点选择错误！") )
            oldPo.code=po.code
            oldPo.name=po.name
            oldPo.parentId=po.parentId 
            oldPo.category=po.category
            oldPo.icon=po.icon
            oldPo.url=po.url
            if type(oldPo.methods)==list: oldPo.methods=",".join(po.methods) 
            oldPo.permissionKey=po.permissionKey
            oldPo.component=po.component
            oldPo.order=po.order
            oldPo.status=po.status
            oldPo.remark=po.remark
            
        return await self.edit(request,pk,po,menuPO,self.getUserId(request),before) 
    async def delete(self,request:Request,pk:int):  
        """
        删除
        """   
        return await self.remove(request,pk,menuPO ) 
     


     
    