from api.view_model.base_view import BaseMethodView, AuthMethodView
from sanic.response import text,raw
from typing import List,Optional
from sanic import  Request
from utils import WechatConfig
from wechatpy import WeChatClient

class wx_base_view(AuthMethodView):
    client :WeChatClient=None
    _config:WechatConfig=None
    def __init__(self,config:WechatConfig) -> None:
        super().__init__()
        self._config=config
        self.client=WeChatClient(config.appid, config.appSecret)        


