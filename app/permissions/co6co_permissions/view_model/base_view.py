from co6co_web_db.view_model import BaseMethodView, Request, BaseDbClsView
from .aop.api_auth import authorized, ctx

from co6co_db_ext.db_operations import DbOperations
from co6co_db_ext.db_utils import db_tools
from ..services import getCurrentUserId, getCurrentUserName
from ..model.pos.right import UserPO
from typing import Optional
from co6co_db_ext.jwt_service import JwtService
from ..services.utils import appHelper

class AbsClsView(BaseDbClsView):
    """
    基础类视图
    """

    @property
    def jwtService(self) -> JwtService:
        if not hasattr(self, "_jwtService") or self._jwtService is None:
            jwt_secret = self.app_config.raw.get("SECRET")
            self._jwtService = JwtService(jwt_secret)
        return self._jwtService

    def create_token(
        self,
        user: Optional[UserPO],
        expire_seconds: int = 86400,
        refresh_expire_seconds: int = 3 * 86400,
    ):
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


class _authView(AbsClsView):
    """
    基础类视图
    """
    @property
    def current_user(self):
        f"""
        获取当前用户信息
        :return: 当前用户信息
        :rtype: {"id": int, "user_name": str, "group_id": int}
        """
        return appHelper.current_user(self.request)

    @property
    def userId(self):
        """
        获取用户ID
        """
        return appHelper.current_user_id(self.request) 

    @property
    def userName(self):
        """
        获取用户ID
        """

        return appHelper.current_user_name(self.request) 

    @property
    def groupId(self):
        """
        获取用户ID
        """
        return appHelper.current_user_group_id(self.request) 

class CtxMethodView(_authView):
    decorators = [ctx]

class AuthMethodView(_authView):
    """
    基础类视图
    """ 
    decorators = [authorized]
    

