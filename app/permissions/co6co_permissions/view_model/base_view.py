from co6co_web_db.view_model import BaseClsView, BaseMethodView, Request, BaseDbView
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


class AbsClsView(BaseClsView):
    """
    基础类视图
    """
    @property
    def jwtService(self) -> JwtService:
        if hasattr(self.request.app, "jwtService") and self.request.app.jwtService is None: 
            secret = self.request.app.config["SECRET"]
            self.request.app.jwtService = JwtService(secret)
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
