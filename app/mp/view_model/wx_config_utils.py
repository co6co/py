

from typing import List,Optional
from sanic import  Request
from utils import WechatConfig
from wechatpy import WeChatClient
from co6co.utils import log
#from wechatpy.enterprise import WeChatClient  #企业号客户端 
def get_wx_configs(request:Request)->List[WechatConfig]: 
    """
    系统中所有的公众号配置
    """
    configs:List[dict]=request.app.config.wx_config   
    result:List[WechatConfig]=[]
    for c in configs: 
        config=WechatConfig()
        config.__dict__.update(c)  
        result.append(config)
    return result

def get_wx_config(request:Request,appid:str)->Optional[WechatConfig]: 
    """
    获取公众号配置
    """
    configs:List[dict]=request.app.config.wx_config  
    filtered:filter= filter(lambda c:c.get("appid")==appid,configs) 
    config=WechatConfig()
    for f in filtered: config.__dict__.update(f)  
    return config

def crate_wx_cliet(request:Request,appid:str) -> WeChatClient:
    """
    创建微信客户端 
    WeChatClient与 微信服务器交换
    """ 
    config:WechatConfig=get_wx_config(request,appid) 
    return crate_wx_cliet_by_config(config)

def crate_wx_cliet_by_config(config:WechatConfig) -> WeChatClient:
    """
    创建微信客户端 
    WeChatClient与 微信服务器交换
    """  
    print(config.appid, config.appSecret)
    return WeChatClient(config.appid, config.appSecret) 