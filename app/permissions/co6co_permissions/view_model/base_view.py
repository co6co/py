from co6co_web_db.view_model import   BaseMethodView, Request, BaseDbClsView
from .aop.api_auth import authorized, ctx

from co6co_db_ext.db_operations import DbOperations
from co6co_db_ext.db_utils import db_tools
from ..services import getCurrentUserId, getCurrentUserName
from ..model.pos.right import UserPO
from typing import Optional
from co6co_db_ext.jwt_service import JwtService

class _view:
    def getUserId(self, request: Request):
        """
        获取用户ID
        """
        return getCurrentUserId(request)

    def getUserName(self, request: Request):
        """
        获取当前用户名
        """
        return getCurrentUserName(request)


class CtxMethodView(BaseMethodView, _view):
    decorators = [ctx]


class AuthMethodView(BaseMethodView, _view):
    """
    token 校验
    api 校验
    """ 
    decorators = [authorized]


class AbsClsView(BaseDbClsView):
    """
    基础类视图
    """
    @property
    def jwtService(self) -> JwtService:
        if not hasattr(self, "_jwtService") or self._jwtService is None: 
            jwt_secret=self.app_config.raw.get("SECRET") 
            self._jwtService = JwtService(jwt_secret)
        return self._jwtService  
    def create_token(self, user: Optional[UserPO], expire_seconds: int = 86400, refresh_expire_seconds: int=3*86400):
        """
        创建token
        :return: token
        :rtype: str
        """

        if user is None:
            return None
        else:
            userAgent = self.request.headers["User-Agent"]
            device = None
            refresh_data = user.crate_jwt_refresh_data(device, userAgent) 
            data = self.jwtService.create_token(
                user.jwt_data,
                refresh_data,
                expire_seconds=expire_seconds,
                refresh_expire_seconds=refresh_expire_seconds,
            )
            return data


class AbsAuthClsView(AbsClsView):
    """
    基础类视图
    """

    decorators = [authorized]

    @property
    def current_user(self):
        f"""
        获取当前用户信息
        :return: 当前用户信息
        :rtype: {"id": int, "user_name": str, "group_id": int}
        """
        if "current_user" in self.request.ctx.__dict__.keys():
            return self.request.ctx.current_user
        else:
            raise Exception("当前用户信息不存在")
    @property
    def userId(self):
        """
        获取用户ID
        """
        if self.current_user:
            userId = int(self.current_user["id"])
            return userId
    @property
    def userName(self):
        """
        获取用户ID
        """
        
        userName = int(self.current_user["user_name"])
        return userName
    @property
    def groupId(self):
        """
        获取用户ID
        """ 
        group_id = int(self.current_user["group_id"])
        return group_id 