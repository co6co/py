from co6co_permissions.view_model .base_view import AuthMethodView, BaseMethodView
from sanic.response import text, raw
from typing import List, Optional
from sanic import Request
from utils import WechatConfig
from wechatpy import WeChatClient
from co6co.utils import log
# from wechatpy.enterprise import WeChatClient  #企业号客户端
from view_model.wx_config_utils import get_wx_config, crate_wx_cliet
from services.cache import wxUserCache
from services.user_services import queryWxUserByDb
from services.bll import BaseBll
import json
from co6co_sanic_ext.utils import JSON_util
import datetime
from services.cache import wxCacheData


class wx_resposne:
    """
    调用微信接口返回
    """
    errcode: int
    errmsg: str


class wx_base_view(BaseMethodView):
    """
    公众号视图基类
    无需要通过本地系统权限验证
    """

    def cteate_wx_client(self, request: Request, appid: str) -> Optional[WeChatClient]:
        """
        创建微信客户端 
        WeChatClient与 微信服务器交换
        """
        return crate_wx_cliet(request, appid)

    def get_wx_config(self, request: Request, appid: str) -> Optional[WechatConfig]:
        """
        获取公众号配置
        """
        return get_wx_config(request, appid)

    def get_wx_configs(self, request: Request) -> List[dict]:
        """
        获取配置中的 [{openid,name}]
        """
        configs: List[dict] = request.app.config.wx_config
        return [{"openId": c.get("appid"), "name": c.get("name")} for c in configs]


class wx_authon_views(wx_base_view, AuthMethodView):
    """
    公众号视图基类
    接口需要认证
    """
    def test():
        print("")

    async def getWxCacheData(self, request: Request, userId: int):
        mgr = wxUserCache(request)
        cacheData = mgr.getwxUserCache(userId)
        if cacheData == None:
            bll = BaseBll()
            log.warn("查询用户对应的微信openId")
            cacheData = bll.run(queryWxUserByDb, bll.session, userId)
            log.warn("查询用户对应的微信openId", cacheData)
            # cacheData = await queryWxUserByDb(self.get_db_session(request), userId)
            if cacheData is None:
                log.warn(f"当前用户{userId}可能未关联微信公众号")
                return None
            mgr.setwxUserCache(userId, cacheData)
        return cacheData

    async def getWxClient(self, request: Request) -> Optional[WeChatClient]:
        return await self.getWxClient2(request, self.getUserId(request))

    async def getWxClient2(self, request: Request, userId: int) -> Optional[WeChatClient]:
        cacheData = await self.getWxCacheData(request, userId)
        return self.cteate_wx_client(request, cacheData.ownAppid)

    async def sendTemplateMessage(self, request: Request, toUser: int, title, *contents: str) -> str | None:
        # 告警类型：{{alarmType.DATA}} 告警说明：{{alarmDesc.DATA}} 告警时间：{{alarmTime.DATA}}
        data = {
            "alarmType": {"value": title},
            "alarmDesc": {"value": contents[0]},
            "alarmTime": {"value": datetime.datetime.now()}
        }
        # date 无法使用默认序列化
        data = JSON_util().encode(data)
        data = json.loads(data)
        cacheData: Optional[wxCacheData] = await self.getWxCacheData(request, toUser)

        if cacheData == None:
            # nonlocal msg
            msg = "删除成功，发布信息的用户未关联微信公众号，未能发送模板消息通知用户"
            return msg
        log.warn("userId,openID", cacheData.openId)
        client = self.cteate_wx_client(request, cacheData.ownAppid)
        jsonData = client.message.send_template(cacheData.openId, "9RhGyY90ZIg4ggd57htoo-0WdnFaPmfQ3Z5krCWz6hI", data=data)
        log.info(f"发送模板消息完成<<:{jsonData}")
        return None
