
from typing import Optional
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Delete
from co6co_db_ext.db_utils import db_tools

from co6co.utils import getRandomStr 
from .base_view import AuthMethodView, BaseMethodView
from .biz_view import AbsAddView, AbsAssociationView, AbsQueryView, AbsPkView, AbsExistView,AuthMethodView

from ..model.pos.right import UserPO, RolePO, UserRolePO, AccountPO
from ..model.enum import user_category
from ..model.filters.user_filter import user_filter
from .aop.right_aop import userRoleChanged 
from .aop.user_aop import AccessTokenChange
from co6co_db_ext.session import transactional


@AccessTokenChange
def accessTokenChange(request: Request, token: str, userPo: UserPO = None):
    return userPo.jwt_data


class user_ass_view(AbsAssociationView):
    routePath = "/association/<userId:int>"

    @property
    def association_sql(self) -> Select:
        subSelect = (
            Select(UserRolePO.roleId, UserRolePO.userId)
            .filter(UserRolePO.userId == self.routeValue)
            .subquery()
        )
        select = (
            Select(
                RolePO.id,
                RolePO.name,
                RolePO.code,
                subSelect.c.userId.label("associatedValue"),
            )
            .outerjoin_from(
                RolePO, subSelect, onclause=subSelect.c.roleId == RolePO.id, full=False
            )
            .order_by(RolePO.id.asc())
        )
        return select

    @property
    def delete_sql(self) -> Delete:
        param = self.get_associationParam()
        return Delete(UserRolePO).filter(
            UserRolePO.userId == self.routeValue, UserRolePO.roleId.in_(param.remove)
        )

    async def create_association_po(
        self, session: AsyncSession, associationed_id: int, *args, **kwargs
    ):
        po = UserRolePO()
        po.roleId = associationed_id
        po.userId = self.routeValue
        return po

    @userRoleChanged 
    async def put(self):
        return await super().put()


class user_exist_view(AbsExistView):
    @property
    def column(self):
        return UserPO.id

    @property
    def exist_condition(self):
        return UserPO.userName == self.routeValue, UserPO.id != self.pkValue


class user_exist_post_view(AbsExistView):
    routePath = "/exist"

    @property
    def message(self, result: bool) -> str:
        return f"用户名{self.routeValue}已存在" if result else "用户不存在"

    async def post(self):
        id = self.json.get("id")
        userName = self.json.get("userName") 
        result = await self.actuator.exist(UserPO.userName == userName, UserPO.id != id)
        return self._response(userName, result)


class user_query_view(AbsQueryView):
    routePath = "/"
    cls = user_filter


class users_view(AbsAddView):
    
    async def get(self ):
        """
        用户下拉框数据
        selectTree :  el-Tree
        """
        select = Select(UserPO.userName.label("name"), UserPO.id).order_by(
            UserPO.id.asc()
        )
        return await self.query_list(  select, isPO=False)

    async def put(self ):
        """
        增加
        """
        po = UserPO()
        userId = self.userId

        async def before(po: UserPO, session: AsyncSession, request):
            exist = await db_tools.exist(
                session, UserPO.userName.__eq__(po.userName), column=UserPO.id
            )
            if exist:
                return response_json(
                    Result.fail(message=f"'{po.userName}'已存在！")
                )
            if po.salt is None:
                po.salt = getRandomStr(6)
            if (
                po.category == user_category.normal.val
                or po.category == user_category.system.val
            ):
                po.password = po.encrypt(po.password)
            else:
                accessTokenChange(self.request, po.password, po)

        return await self.add( po, userId=userId, beforeFun=before)


class user_view(AbsPkView):
    async def put(self ):
        """
        编辑
        """ 
        async def before(oldPo: UserPO, po: UserPO, session: AsyncSession, request):
            exist = await db_tools.exist(
                session,
                UserPO.id != oldPo.id,
                UserPO.userName.__eq__(po.userName),
                column=UserPO.id,
            )
            if exist:
                return response_json( Result.fail(message=f"'{po.userName}'已存在！") ) 
        return await self.edit(  self.pkValue, UserPO, userId=self.userId, fun=before )

    async def delete(self):
        """
        删除
        """
        if self.pkValue == 1:
            return response_json(Result.fail(message="不能删除系统默认用户！"))

        async def before(po: UserPO, session: AsyncSession):
            # 用户角色关联
            count = await db_tools.count(
                session, UserRolePO.userId == po.id, column=UserRolePO.userId
            )
            if count > 0:
                return response_json(
                    Result.fail(
                        message=f"该'{po.userName}'用户关联有‘{count}’角色，不能删除！"
                    )
                )
            count = await db_tools.count(
                session, AccountPO.userId == po.id, column=AccountPO.uid
            )
            if count > 0:
                return response_json(
                    Result.fail(
                        message=f"该'{po.userName}'用户关联有‘{count}’账号，不能删除！"
                    )
                )

        return await self.remove(  self.pkValue, UserPO, beforeFun=before)


class sys_users_view(AuthMethodView):
    routePath = "/reset"
    
    async def post(self ):
        """
        重置密码
        """
        data = self.json
        userName = data["userName"]
        password = data["password"]
        select = Select(UserPO).filter(UserPO.userName == userName)
        if userName == None or password == None or len(password) < 6:
            return response_json(Result.fail(message="请检查提交的用户和密码！"))

        async def edit(_, one: Optional[UserPO]):
            if one is not None:
                if one.salt is None:
                    return response_json(
                        Result.fail(
                            message=f"用户[{userName}],通过关联创建的用户，完善信息才能重置密码"
                        )
                    )
                if (
                    one.category == user_category.normal.val
                    or one.category == user_category.system.val
                ):
                    one.password = one.encrypt(password)
                else:
                    one.password = password
                    accessTokenChange(self.request, password, one)
                return response_json(Result.success())
            else:
                return response_json(
                    Result.fail(
                        message=f"所提供的用户名[{userName}]不存在，请刷新重试！"
                    )
                )

        return await self.update_one( select, edit)


class ticketView(AuthMethodView ):
    routePath = "/ticket/<code:str>"
    
    async def get(self):
        """
        通过 code 换取 token
        code 为临时 有过期时间
        """
        code=self.match_info["code"] 
        data=self.jwtService.decode(code)
        
        userId=data["id"]
        if userId is  None:
            return response_json(Result.fail(message="code 无效或已过期"))
        select = Select(UserPO).filter(UserPO.id.__eq__(userId))
        user: Optional[UserPO] = await self.actuator.query_one_entity(select)  
        if user is not None:
            token = await self.jwtService.create_token(user.jwt_data, user.crate_jwt_refresh_data("deviceId",self.request.headers.get("user-agent")))
            return response_json(
                Result.success(data=token, message="票据登录成功")
            )
        else:
            return response_json(Result.fail(message="未找到所属用户"))
