
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Delete
from co6co_db_ext.db_utils import db_tools 
from co6co_db_ext.session import transactional

 
from .biz_view import AbsExistView,AbsAssociationView,AuthMethodView,AbsPkView
from ..model.pos.right import UserGroupPO, RolePO, UserGroupRolePO
from ..model.filters.user_group_filter import user_group_filter
from co6co.utils import log
from .aop.right_aop import userRoleChanged

 
class user_group_exist_view(AbsExistView):
    @property
    def column(self):
        return UserGroupPO.id
    @property 
    def exist_condition(self)  :
        return  UserGroupPO.code == self.param_code, UserGroupPO.id != self.param_pk 



class user_group_ass_view(AbsAssociationView):
    routePath = "/association/<userGroupId:int>" 
    @property 
    def association_sql(self)->Select :
        subSelect = Select(UserGroupRolePO.roleId, UserGroupRolePO.userGroupId).filter(
            UserGroupRolePO.userGroupId == self.routeValue).subquery()
        select = (
            Select(RolePO.id, RolePO.name, RolePO.code,
                   subSelect.c.roleId.label("associatedValue"))
            .outerjoin_from(RolePO, subSelect, onclause=subSelect.c.roleId == RolePO.id, full=False)
            .order_by(RolePO.name.asc())
        )
        return select 
    @property 
    def delete_sql(self) -> Delete:
        param = self.get_associationParam()
        return   Delete(UserGroupRolePO).filter(UserGroupRolePO.userGroupId ==
                                           self.routePath, UserGroupRolePO.roleId .in_(param.remove))
    async def create_association_po(self,session:AsyncSession,associationed_id:int,*args,**kwargs)  :
        po = UserGroupRolePO()
        po.roleId = associationed_id
        po.userGroupId = self.routeValue
        return po

    @userRoleChanged
    @transactional
    async def put(self):
        return await super().put()
 
class user_groups_tree_view(AuthMethodView):
    routePath = "/tree" 
    async def get(self, parendId:int=None ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(UserGroupPO.id, UserGroupPO.name,
                   UserGroupPO.code, UserGroupPO.parentId)
            .order_by(UserGroupPO.parentId.asc())
        )
        return await self.query_tree( select, rootValue=parendId,  pid_field='parentId', id_field="id", isPO=False)
    @transactional
    async def post(self ):
        """
        树形 table数据
        tree 形状 table
        """
        param = user_group_filter()
        param.__dict__.update(self.json)
        if len(param.filter()) > 0:
            return await self.query_page( param)
        return await self.query_tree( param.create_List_select(), rootValue=0, pid_field='parentId', id_field="id")


class user_groups_sub_tree_view(user_groups_tree_view):
    routePath = "/tree/<parendId:int>"
    @property
    def parendId(self):
        return self.match_info.get("parendId")
     
    async def get(self ):
        """
        返回子 树形选择下拉框数据
        """
        return await super().get(self.parendId)


class user_groups_view(AuthMethodView): 
    async def get(self ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(UserGroupPO.id, UserGroupPO.name,
                   UserGroupPO.code, UserGroupPO.parentId)
            .order_by(UserGroupPO.parentId.asc())
        )
        return await self.query_list( select,  isPO=False)

    
    async def post(self ):
        """
        树形 table数据
        tree 形状 table
        """
        param = user_group_filter()
        param.__dict__.update(self.json)
        return await self.query_list( param.list_select)

    async def put(self ):
        """
        增加
        """
        po = UserGroupPO()
        userId = self.userId

        async def before(po: UserGroupPO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  UserGroupPO.code.__eq__(po.code), column=UserGroupPO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.add( po, userId=userId, beforeFun=before) 


class user_group_view(AbsPkView):
   
    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: UserGroupPO, po: UserGroupPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, UserGroupPO.id != oldPo.id, UserGroupPO.code.__eq__(po.code), column=UserGroupPO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))
            if po.parentId == oldPo.id:
                return response_json(Result.fail(message=f"'父节点选择错误！"))

        return await self.edit( self.routeValue, UserGroupPO, userId=self.userId, fun=before)

    async def delete(self ):
        """
        删除
        """
        async def before(po: UserGroupPO, session: AsyncSession):
            count = await db_tools.count(session, UserGroupPO.parentId == po.id, column=UserGroupPO.id)
            if count > 0:
                return response_json(Result.fail(message=f"该'{po.name}'节点下有‘{count}’节点，不能删除！"))
        return await self.remove(  self.routeValue, UserGroupPO, beforeFun=before)
