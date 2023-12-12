from functools import wraps
from sanic.request import Request
from view_model.wx import crate_wx_cliet,get_wx_config
from wechatpy.oauth import WeChatOAuth
from sanic.response import text,raw,redirect
from co6co.utils import log

class Authon_param:
    code: str
    appid: str
    url: str
    scope:str #snsapi_base|snsapi_userinfo
    state:str 


def oauth(method):
    """
    微信页面 认证
    """
    @wraps(method)
    def warpper(request:Request,param:Authon_param ): 
        config=get_wx_config(request,param.appid) 
        oauth=WeChatOAuth(param.appid, config.appSecret, param.url, scope=param.scope, state=param.state) 
        oauth .check_access_token()
        oauth.fetch_access_token(param.code)
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