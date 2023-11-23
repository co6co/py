from wechatpy import WeChatClient
from wechatpy.client.api.base import BaseWeChatAPI
import inspect

client = WeChatClient('appid', 'secret')
# 发送图片消息
res = client.message.send_image('openid', 'media_id')
# 查询自定义菜单
menu = client.menu.get()


def _is_api_endpoint(obj):
    return isinstance(obj, BaseWeChatAPI)
class BaseWeChatClient:
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        api_endpoints = inspect.getmembers(self, _is_api_endpoint)
        for name, api in api_endpoints:
            api_cls = type(api)
            api = api_cls(self)
            setattr(self, name, api)
        return self
    
class WeChatMenu(BaseWeChatAPI):
    def get(self):
        return self

class WeChatClient三(BaseWeChatClient):
    API_BASE_URL = "https://api.weixin.qq.com/cgi-bin/" 
    menu = WeChatMenu()


