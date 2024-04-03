from view_model.base_view import BaseMethodView, AuthMethodView
from sanic.response import text,raw
from typing import List,Optional
from sanic import  Request
from utils import WechatConfig
from wechatpy import WeChatClient
from co6co.utils import log
#from wechatpy.enterprise import WeChatClient  #企业号客户端
from view_model.wx_config_utils import get_wx_config,crate_wx_cliet

class wx_resposne:
    """
    调用微信接口返回
    """
    errcode:int
    errmsg:str

class wx_base_view(BaseMethodView): 
    """
    公众号视图基类
    无需要通过本地系统权限验证
    """
    def cteate_wx_client(self,request:Request,appid:str) -> WeChatClient:
        """
        创建微信客户端 
        WeChatClient与 微信服务器交换
        """ 
        return crate_wx_cliet(request,appid)
     
    def get_wx_config(self,request:Request,appid:str)->Optional[ WechatConfig]: 
        """
        获取公众号配置
        """
        return get_wx_config(request,appid)
    
    def get_wx_configs(self,request:Request)->List[dict]: 
        """
        获取配置中的 [{openid,name}]
        """
        configs:List[dict]=request.app.config.wx_config   
        return [{"openId":c.get("appid"),"name":c.get("name")} for c in configs]

class wx_authon_views(wx_base_view,AuthMethodView):
    """
    公众号视图基类
    接口需要认证
    """
    def test():
        print("")



