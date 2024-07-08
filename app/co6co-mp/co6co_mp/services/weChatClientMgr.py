from __future__ import annotations
from wechatpy import WeChatClient
from wechatpy.session.memorystorage import MemoryStorage
from wechatpy.session.memcachedstorage import MemcachedStorage
from wechatpy.session.redisstorage import RedisStorage
from wechatpy.session.shovestorage import ShoveStorage
from co6co_permissions.model.pos.right import UserGroupPO

from model import WechatConfig
from typing import Dict, Tuple
# 定义全局的 WeChatClient 实例
_wechatClientMgr: ManagewechatClients = None


class wechatClientModel:
    config: WechatConfig
    client: WeChatClient

    def __init__(self, config: WechatConfig, client: WeChatClient) -> None:
        self.config = config
        self.client = WeChatClient
        pass


class ManagewechatClients:
    maps: Dict[str, wechatClientModel] = {}

    def __init__(self) -> None:
        self.maps = {}
        pass

    @staticmethod
    @property
    def sigleInstance() -> ManagewechatClients:
        global _wechatClientMgr
        if _wechatClientMgr is None:
            _wechatClientMgr = ManagewechatClients()
        return _wechatClientMgr

    def addClient(self, config: WechatConfig, session_storage: MemoryStorage | MemcachedStorage | RedisStorage | ShoveStorage = None):
        """
        @param session_storage:  能自动更新 jsapi_ticket 和 access_token
            1. 简单存储方式 (默认)
               from wechatpy.session.memorystorage import MemoryStorage
               session_storage = MemoryStorage()
            2. pymemcache 分布式的内存对象缓存系统
                from wechatpy.session.memcachedstorage import MemcachedStorage
                from pymemcache.client import base
                # 创建Memcached客户端实例
                memcached_client = base.Client(('localhost', 11211))
                # 使用Memcached客户端创建MemcachedStorage实例
                session_storage = MemcachedStorage(memcached_client)
            3. Redis
                redis = Redis(host='localhost', port=6379)
                session_storage = RedisStorage(redis)
        return WeChatClient
        """
        if config.appid not in self.maps:
            if session_storage == None:
                session_storage = MemoryStorage()
            wechat_client = WeChatClient(
                config.appid, config.appSecret, session=session_storage)
            self.maps.update(
                {config.appid: wechatClientModel(config, wechat_client)})
            return wechat_client
        else:
            _, client = self.get(config.appid)
            return client

    def get(self, appid: str) -> Tuple[WechatConfig | None, WeChatClient | None]:
        """
        获取公众号配置和公众号客户端
        """
        if appid in self.maps:
            inistance = self.maps.get(appid)
            return inistance.config, inistance.client
        else:
            return None, None

    def remove(self, appid: str) -> wechatClientModel | None:
        """
        移除公众号客户端
        """
        if appid in self.maps:
            # popitem 随机移除
            # del self.maps[appid]
            value = self.maps.pop(appid, None)
            return value
        return None
