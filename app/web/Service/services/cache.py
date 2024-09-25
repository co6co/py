
from sanic import Request
from co6co_db_ext.res.result import Result
from co6co_db_ext.db_utils import QueryOneCallable
from model.enum import Account_category
from model.wxModel import wxCacheData
from co6co_permissions.model.pos.right import AccountPO
from co6co_permissions.services import getSecret, generateCode
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import Select
from co6co.utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from model.pos.wx import WxUserPO
from co6co_web_db.services.cacheManage import CacheManage
from co6co_permissions.services import getCurrentUserId


class wxUserCache(CacheManage):
    """
    不能在多请求中使用，即： request 必须是同一个不然会可能出現數據异常
    """
    userId: int

    def __init__(self, request: Request) -> None:
        # 微信认证中 userid可能为挂在上去
        self.userId = getCurrentUserId(request)
        super().__init__(request.app)

    @property
    def wxUserKey(self):
        return f'wx_data_{self.userId}'

    def setwxUserCache(self, userId: int, data: wxCacheData):
        if userId != None:
            self.userId = userId

        self.setCache(self.wxUserKey, data)

    def getwxUserCache(self, userId: int) -> wxCacheData | None:
        if userId != None:
            self.userId = userId
        return self.getCache(self.wxUserKey)

    def clearwxUserCache(self, userId: int = None) -> wxCacheData | None:
        if userId != None:
            self.userId = userId
        return self.clearwxUserCache(self.wxUserKey)
