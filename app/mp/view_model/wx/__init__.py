from view_model.base_view import BaseMethodView, AuthMethodView
from sanic.response import text,raw
from typing import List,Optional
from sanic import  Request
from utils import WechatConfig
from wechatpy import WeChatClient

class wx_base_view(AuthMethodView): 
    """
    公众号视图基类
    需要通过权限验证
    """
    def cteate_wx_client(self,request:Request,appid:str) -> WeChatClient:
        """
        创建微信客户端 
        WeChatClient与 微信服务器交换
        """ 
        config=self.get_wx_config(request,appid)
        return WeChatClient(config.appid, config.appSecret) 
     
    def get_wx_config(self,request:Request,appid:str)->Optional[ WechatConfig]: 
        """
        获取公众号配置
        """
        configs:List[dict]=request.app.config.wx_config  
        filtered:filter= filter(lambda c:c.get("appid")==appid,configs) 
        config=WechatConfig()
        for f in filtered: config.__dict__.update(f)  
        return config   
    def get_wx_configs(self,request:Request)->List[dict]: 
        """
        获取配置中的 [{openid,name}]
        """
        configs:List[dict]=request.app.config.wx_config   
        return [ {"openId":c.get("appid"),"name":c.get("name")} for c in configs]


