from functools import wraps
from sanic.request import Request
from view_model.wx import crate_wx_cliet,get_wx_config
from wechatpy.oauth import WeChatOAuth
from sanic.response import text,raw,redirect
from co6co.utils import log
from model.enum.wx import wx_authon_scope
from model.pos .wx import    WxUserPO
from model.pos .right import UserPO


class Authon_param:
    code: str
    appid: str
    url: str
    scope:wx_authon_scope #snsapi_base|snsapi_userinfo
    state:str

    def __init__(self,appid:str) -> None:
        self.scope=wx_authon_scope.snsapi_userinfo
        self.url="/"
        self.appid=appid
     
    def setState(self, state:str) :
        """
        随机码-0|1-url|随机码
        arr[0] #随机码
        arr[1] #scope 类型：0|1
        arr[2] #url
        arr[3] #随机码
        """
        self.state=state
        arr=state.split("-")
        if len(arr)==4:
            #arr[0] #随机码
            self.scope=wx_authon_scope.snsapi_base if arr[1]==wx_authon_scope.snsapi_base.val else wx_authon_scope.snsapi_userinfo
            self.url=arr[2] #url
            #arr[3] #随机码

    def __repr__(self) -> str:
        return f"class<Authon_param> ==>code:{self.code},appid:{self.appid},url:{self.url},scope:{self.scope},state:{self.state}" 

       

        






def oauth(method):
    """
    微信页面 认证
    """
    @wraps(method)
    def warpper(request:Request,param:Authon_param ): 
        config=get_wx_config(request,param.appid) 
        # 这样需要优化 ，仅第一次需要调用其他需要调用刷新
        oauth=WeChatOAuth(param.appid, config.appSecret, param.url, scope=param.scope.key, state=param.state) 
        #oauth .check_access_token()
        # 第二步 通过code换取网页授权access_token
        data=oauth.fetch_access_token(param.code)
        log.warn(param.appid)
        log.warn(data)

        # 第三步 刷新 access_token
        #access_token拥有较短的有效期，当access_token超时后，
        # 可以使用refresh_token进行刷新，refresh_token有效期为30天，当refresh_token失效之后，需要用户重新授权。
        #oauth.refresh_access_token(data.refresh_token)


        # 第四步 获取用户信息
        wxUser=oauth.get_user_info(  ) 
        log.warn(f"获取的wxUser:{wxUser}") 
        '''
        与数据库做关联操作
        '''
        return redirect( param.url)
        return method(*args, **kwargs)
        '''
        authorize_url:'https://open.weixin.qq.com/connect/oauth2/authorize?appid=123456&redirect_uri=http://localhost'
            '&response_type=code&scope=snsapi_base#wechat_redirect'
        authorize_url:'https://open.weixin.qq.com/connect/oauth2/authorize?appid=123456&redirect_uri=/index.html
                    &response_type=code&scope=snsapi_userinfo&state=stateCode#wechat_redirect
        qrconnect_url: 'https://open.weixin.qq.com/connect/qrconnect?appid=123456&redirect_uri=http://localhost'
            '&response_type=code&scope=snsapi_login#wechat_redirect'
        ''' 
        return method(*args, **kwargs)
    return warpper