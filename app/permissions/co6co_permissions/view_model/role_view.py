
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Delete

from co6co_db_ext.db_utils import db_tools 
from co6co_web_db.model.params import associationParam

from datetime import datetime
from .base_view import AuthMethodView,AuthMethodView
from .biz_view import AbsExistView,AbsAssociationView,AbsSelectView,AbsPkView,AbsQueryView,AbsAddView

from ..model.pos.right import RolePO, MenuRolePO, UserRolePO, UserGroupRolePO, menuPO
from ..model.filters.role_filter import role_filter
from .aop.right_aop import userRoleChanged 

class roles_ass_exist_view(AbsExistView):
    @property
    def column(self):
        return RolePO.id
    @property 
    def exist_condition(self)  :
        return  RolePO.code == self.param_code, RolePO.id != self.param_pk 


class roles_ass_view(AbsAssociationView):
    routePath = "/association/<roleId:int>"
    @property
    def is_tree(self):
        return True 
    @property 
    def association_sql(self)->Select :
        subSelect = Select(MenuRolePO.menuId, MenuRolePO.roleId).filter(
            MenuRolePO.roleId == self.routeValue).subquery()
        select = (
            Select(menuPO.id, menuPO.name, menuPO.code, menuPO.parentId,
                   subSelect.c.roleId.label("associatedValue"))
            .outerjoin_from(menuPO, subSelect, onclause=subSelect.c.menuId == menuPO.id, full=False)
            .order_by(menuPO.parentId.asc())
        )
        return select 
    @property 
    def delete_sql(self) -> Delete:
        param = self.get_associationParam()
        return  Delete(MenuRolePO).filter(MenuRolePO.roleId == self.routeValue, MenuRolePO.menuId .in_(param.remove))
    async def create_association_po(self,session:AsyncSession,associationed_id:int,*args,**kwargs)  :
        po = MenuRolePO()
        po.menuId = associationed_id
        po.roleId = self.routeValue
        return po

    @userRoleChanged
    async def put(self):
        return await super().put()

class roles_query_view(AbsQueryView):
    """
     table数据 
    """
    routePath = "/" # 未兼容UI

    cls = role_filter


class roles_view(AuthMethodView):
    async def get(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(RolePO.id, RolePO.name, RolePO.code)
            .order_by(RolePO.order.asc())
        )
        return await self.query_list( select,  isPO=False)  
    async def put(self ):
        """
        增加
        """
        po = RolePO()
        userId =self.userId 
        async def before(po: RolePO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  RolePO.code.__eq__(po.code), column=RolePO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.add( po, userId=userId, beforeFun=before) 


class role_view(AbsPkView): 

    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: RolePO, po: RolePO, session: AsyncSession, request):
            exist = await db_tools.exist(session, RolePO.id != oldPo.id, RolePO.code.__eq__(po.code), column=RolePO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))

        return await self.edit( self.routeValue, RolePO, userId=self.userId, fun=before)

    async def delete(self ):
        """
        删除
        """
        async def before(po: RolePO, session: AsyncSession):
            count = await db_tools.count(session, MenuRolePO.roleId == po.id, column=RolePO.id)
            if count > 0:
                return response_json(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个菜单，不能删除！"))
            count = await db_tools.count(session, UserGroupRolePO.roleId == po.id, column=RolePO.id)
            if count > 0:
                return response_json(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个用户组，不能删除！"))
            count = await db_tools.count(session, UserRolePO.roleId == po.id, column=RolePO.id)
            if count > 0:
                return response_json(Result.fail(message=f"该'{po.name}'角色关联了‘{count}’个用户，不能删除！"))

        return await self.remove( self.routeValue, RolePO, beforeFun=before)
