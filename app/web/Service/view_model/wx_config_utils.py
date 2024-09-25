

from typing import List, Optional
from sanic import Request
from utils import WechatConfig
from wechatpy import WeChatClient
from co6co.utils import log
# from wechatpy.enterprise import WeChatClient  #企业号客户端
from services.weChatClientMgr import ManagewechatClients


def get_wx_configs(request: Request) -> List[WechatConfig]:
    """
    系统中所有的公众号配置
    """
    configs: List[dict] = request.app.config.wx_config
    result: List[WechatConfig] = []
    for c in configs:
        config = WechatConfig()
        config.__dict__.update(c)
        result.append(config)
    return result


def get_wx_config(request: Request, appid: str) -> Optional[WechatConfig]:
    """
    获取公众号配置
    """
    configs: List[dict] = request.app.config.wx_config
    filtered: filter = filter(lambda c: c.get("appid") == appid, configs)

    for f in filtered:
        config = WechatConfig()
        config.__dict__.update(f)
        return config
    return None


def crate_wx_cliet(request: Request, appid: str) -> Optional[WeChatClient]:
    """
    创建微信客户端 
    WeChatClient与 微信服务器交换
    """
    config: Optional[WechatConfig] = get_wx_config(request, appid)
    if config != None:
        return get_wx_cliet_by_config(config)
    return None


def get_wx_cliet_by_config(config: WechatConfig) -> WeChatClient:
    """
    获取微信客户端
    1. 查找是否存在，存在返回客户端
    2. 不存在根据配置创建
    WeChatClient与 微信服务器交换
    """
    log.warn("get Client...")
    _, client, = ManagewechatClients.sigleInstance().get(config.appid)

    if client == None:
        client = ManagewechatClients.sigleInstance().createClient(config)

    log.warn("access_tonen", client.access_token, type(client))
    return client
