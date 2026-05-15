from __future__ import annotations
from wechatpy import WeChatClient
from wechatpy.session.memorystorage import MemoryStorage
from wechatpy.session.memcachedstorage import MemcachedStorage
from wechatpy.session.redisstorage import RedisStorage
from wechatpy.session.shovestorage import ShoveStorage

from typing import Dict, Tuple, List

class WechatConfig: 
    def __init__(self) -> None:
        self.name: str = None  # 仅为方便查看
        self.appid: str
        self.appSecret: str
        self.token: str =None  #    token
        self.encodingAESKey: str = None  # 公众号的secret
        self.encrypt_mode: str = None   ## 可选项：normal/compatible/safe，分别对应于 明文/兼容/安全 模式
        self.tamplate:List=[] #{id:str ,模板消息ID,url:str模板消息URL} 

     
    
    def init(self, config: Dict):
        self.appid = config.get("appid")
        self.appSecret = config.get("appSecret")
        self.token=config.get('token')
        self.encodingAESKey=config.get("encodingAESKey")
        self.encrypt_mode=config.get("encrypt_mode")
        self.tamplate=config.get("tamplate",[]) 
        
    @staticmethod
    def get_config(appid: str, all_config: List[WechatConfig|Dict]) -> WechatConfig | None:
        for config in all_config:
            if isinstance(config, Dict):
                config2 = WechatConfig()
                config2.init(config)
                config=config2
            if config.appid == appid:  # ✅ 修复：使用 dataclass 属性访问
                return config
        return None
    


class WechatClientModel:  # ✅ 修复：PascalCase 命名
    config: WechatConfig
    client: WeChatClient

    def __init__(self, config: WechatConfig, client: WeChatClient) -> None:
        self.config = config
        self.client = client  # ✅ 修复：使用传入的 client 参数


class ManageClient:
    _instance: ManageClient = None

    def __init__(self) -> None:
        self.maps: Dict[str, WechatClientModel] = {}

    @classmethod  # ✅ 修复：使用 classmethod
    def get_instance(cls) -> ManageClient:
        if cls._instance is None:
            cls._instance = ManageClient()
        return cls._instance

    def create_get_client(  # ✅ 修复：snake_case 命名
        self,
        config: WechatConfig,
        session_storage: MemoryStorage
        | MemcachedStorage
        | RedisStorage
        | ShoveStorage = None,
    ):
        """
        创建客户端，如果客户端不存在则创建，否则返回已存在的客户端。
        @param config: 客户端配置信息
        @return: WeChatClient 客户端实例

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
            if session_storage is None:
                session_storage = MemoryStorage()
            wechat_client = WeChatClient(
                config.appid, config.appSecret, session=session_storage
            )
            self.maps[config.appid] = WechatClientModel(config, wechat_client)
            return wechat_client
        else:
            _, client = self.get(config.appid)
            return client

    def get(self, appid: str) -> Tuple[WechatConfig | None, WeChatClient | None]:
        if appid in self.maps:
            instance = self.maps.get(appid)
            return instance.config, instance.client
        return None, None

    def remove(self, appid: str) -> WechatClientModel | None:
        return self.maps.pop(appid, None)
