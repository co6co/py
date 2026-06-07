
from sanic.response import text 

from sqlalchemy.sql import Select, or_, and_, text as sqlText
from co6co_db_ext.db_utils import db_tools 
from ..base_view import CtxMethodView
from ...model.enum import menu_type
from ...model.pos.right import menuPO, MenuRolePO, UserPO, UserGroupRolePO, UserRolePO
from ...model.filters.menu_filter import menu_filter 


class ui_tree_view(CtxMethodView):
    routePath = "/tree/" 
    async def get(self ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        queryRoleSml = sqlText(
            """
            (select  g.role_id from sys_user_group_role g
            INNER JOIN sys_user u
            on g.user_group_id=u.user_group_id
            where u.id=:userId)
            UNION
            (select ur.role_id from sys_user_role ur
            INNER JOIN sys_user u
            on ur.user_id=u.id
            where u.id=:userId)
            """
        )
        currentId =self.userId
        userRolesSelect = (
            Select(UserRolePO.roleId).filter(UserRolePO.userId == UserPO.id, UserPO.id == currentId)
        )
        userGroupRolesSelect = (
            Select(UserGroupRolePO.roleId).filter(UserGroupRolePO.userGroupId == UserPO.userGroupId, UserPO.id == currentId)
        )
        # roleList=await self._query(request,queryRoleSml,isPO=False,param={"userId":1})
        queryRoleSelect =  userRolesSelect.union(userGroupRolesSelect)
        #roleList=await self._query(request,queryRoleSml,isPO=False,param={"userId":1}) 
        session=self.db_session
        from co6co_db_ext.session import session_context
        roleList=[]
        roleList = await db_tools.execForMappings(session,queryRoleSelect ,queryOne=False) 
        #log.warn(roleList)
        roleList = [d.get("role_id") for d in roleList]
        select = (
            Select(menuPO.id, menuPO.category, menuPO.parentId, menuPO.name, menuPO.code, menuPO.icon,  menuPO.url, menuPO.component, menuPO.permissionKey, menuPO.methods)
            .join(MenuRolePO, onclause=MenuRolePO.menuId == menuPO.id)
            .filter(and_(or_(menuPO.category.__eq__(menu_type.group.val), menuPO.category.__eq__(menu_type.subView.val), menuPO.category.__eq__(menu_type.view.val), menuPO.category.__eq__(menu_type.button.val)), MenuRolePO.roleId.in_(roleList)))
            .order_by(menuPO.parentId.asc(), menuPO.order.asc())
        ).distinct(menuPO.id)

        return await self.query_tree( select, rootValue=0, pid_field='parentId', id_field="id", isPO=False)

    async def post(self ):
        """
        树形 table数据
        tree 形状 table
        """
        param = menu_filter()
        param.__dict__.update(self.json)
        if len(param.filter()) > 0:
            return await self.query_page( param)
        return await self.query_tree( param.create_List_select(), rootValue=0, pid_field='parentId', id_field="id")
