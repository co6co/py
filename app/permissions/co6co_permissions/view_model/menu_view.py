
from numpy import insert
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools 
from .biz_view import AbsExistView,AuthMethodView, AbsPkView
 
from ..model.pos.right import menuPO
from ..model.filters.menu_filter import menu_filter
from .aop.right_aop import menuChanged
 
class menu_exist_view(AbsExistView):
    @property
    def column(self):
        return menuPO.id
    @property 
    def exist_condition(self)  :
        return  menuPO.code == self.param_code, menuPO.id != self.param_pk 

class menu_tree_view(AuthMethodView):
    routePath = "/tree"

    async def get(self ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(menuPO.id, menuPO.name, menuPO.code, menuPO.parentId)
            .order_by(menuPO.order.asc())
        )
        return await self.query_tree( select, pid_field='parentId', id_field="id", isPO=False)

    async def post(self ):
        """
        树形 table数据
        没有条件 返回   tree_data
        有条件 返回     PAGED_list 

        """
        param = menu_filter()
        param.__dict__.update(self.json)
        if len(param.filter()) > 0:
            return await self.query_page( param)
        return await self.query_tree( param.create_List_select().order_by(menuPO.order.asc()), rootValue=0, pid_field='parentId', id_field="id")


class menus_view(AuthMethodView):
    async def get(self):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(menuPO.id, menuPO.name, menuPO.code, menuPO.parentId)
            .order_by(menuPO.parentId.asc())
        )
        return await self.query_list( select,  isPO=False)

    async def post(self):
        """
        树形 table数据
        tree 形状 table
        """
        param = menu_filter()
        param.__dict__.update(self.json)

        return await self.query_list( param.list_select)

    @menuChanged
    async def put(self):
        """
        增加
        """
        po = menuPO()
        userId = self.userId
        po.__dict__.update(self.json)
        if isinstance(po.methods, list):
            po.methods = ",".join(po.methods)

        async def before(po: menuPO, session: AsyncSession, *args, **kwargs):
            exist = await db_tools.exist(session,  menuPO.code.__eq__(po.code), column=menuPO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")

        return await self.add( po, json2Po=False, userId=userId, beforeFun=before) 


class menu_view(AbsPkView): 

    async def get(self):
        select = (
            Select(menuPO.id, menuPO.name, menuPO.component, menuPO.category, menuPO.code, menuPO.status)
            .filter(menuPO.id.__eq__(self.routeValue))
        )
        return await self.get_one( select,  isPO=False)

    @menuChanged
    async def put(self):
        """
        编辑
        """
        po = menuPO()
        po.__dict__.update(self.json)
        if isinstance(po.methods, list):
            po.methods = ",".join(po.methods)

        async def before(oldPo: menuPO, po: menuPO, session: AsyncSession, *args, **kwargs):
            exist = await db_tools.exist(session, menuPO.id != oldPo.id, menuPO.code.__eq__(po.code), column=menuPO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")
            if po.parentId == oldPo.id:
                return Result.fail(message="'父节点选择错误！")
        return await self.edit(self.routeValue, menuPO, po=po, userId=self.userId, fun=before)

    @menuChanged
    async def delete(self):
        """
        删除
        """
        async def before(po: menuPO, session: AsyncSession,*args, **kwargs):
            count = await db_tools.count(session, menuPO.parentId == po.id, column=menuPO.id)
            if count > 0:
                return Result.fail(message=f"该'{po.name}'节点下有‘{count}’节点，不能删除！")
        return await self.remove( self.routeValue, menuPO, beforeFun=before)


class menu_batch_view(AuthMethodView):
    routePath = "/batch"

    @menuChanged
    async def post(self ):
        """
        批量增加
        """
        data = self.json 
        async def before(po: menuPO, session: AsyncSession, *args, **kwargs):
            exist = await db_tools.exist(session,  menuPO.code.__eq__(po.code), column=menuPO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")
        if isinstance(data, list):
            polist = []
            userId = self.userId
            for js in data:
                po = menuPO()
                po.__dict__.update(js)
                polist.append(po)
            return await self.batchAdd( polist,   userId=userId, beforeFun=before)
        return self.response_json(Result.fail(message="json 数据不是列表"))
