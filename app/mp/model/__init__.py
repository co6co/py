class WechatConfig:
    """
    配置信息[开发信息| 服务器配置]
    开发信息:appid,appSecret
    服务器配置:url, token,encodingAESKey,encrypt_mode
    
    """
    name:str=None, #仅为方便查看
    appid:str=None,		                #    公众号的appid
    appSecret:str=None,
    
    token:str=None, 		            #    token
    encodingAESKey:str=None,		            # 公众号的secret
    encrypt_mode:str=None   ## 可选项：normal/compatible/safe，分别对应于 明文/兼容/安全 模式
    def __init__(self) -> None:
        pass

