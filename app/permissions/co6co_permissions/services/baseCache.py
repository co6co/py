from sanic import Request
from co6co_db_ext.cacheManage import CacheManage 
from .utils import appHelper


class CustomSanicCache(CacheManage): 
    def __init__(self, request: Request) -> None:
        cache, session, service = appHelper.get_app_param(request)
        self.request = request
        super().__init__(cache, session=session, db_service=service)

    @property
    def userId(self):
        """
        当前用户ID
        """
        # 微信认证中 userid可能未挂在上去
        return appHelper.current_user_id(self.request) 