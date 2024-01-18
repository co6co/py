from functools import wraps
from sanic.request import Request
from services import  getAccountuuid 
from view_model.wx_config_utils import get_wx_config
from wechatpy.oauth import WeChatOAuth
from sanic.response import text,raw,redirect
from co6co.utils import log
from model.enum.wx import wx_authon_scope
from model.wx import wxUser
from model.pos .wx import    WxUserPO
from model.pos .right import UserPO

from services.wx_services import oauth_wx_user_to_db
import asyncio 

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
            self.url=arr[2].replace("**","#") #url 
            #arr[3] #随机码 
    def __repr__(self) -> str:
        return f"class<Authon_param> ==>code:{self.code},appid:{self.appid},url:{self.url},scope:{self.scope},state:{self.state}" 




def oauth(method):
    """
    微信页面 认证
    """
    @wraps(method)
    async def  warpper(request:Request,param:Authon_param ):  
        try: 
            config=get_wx_config(request,param.appid) 
            # 这样需要优化 ，仅第一次需要调用其他需要调用刷新
            oauth=WeChatOAuth(param.appid, config.appSecret, param.url, scope=param.scope.key, state=param.state) 
            #oauth .check_access_token()
            # 第二步 通过code换取网页授权access_token
            data=oauth.fetch_access_token(param.code)  
            log.info(f"access_token:{data}")

            # 第三步 刷新 access_token
            #access_token拥有较短的有效期，当access_token超时后，
            # 可以使用refresh_token进行刷新，refresh_token有效期为30天，当refresh_token失效之后，需要用户重新授权。
            #oauth.refresh_access_token(data.refresh_token)
            return await get_snsapi_userinfo(oauth,request,param)
        except Exception as e:
            log.warn(f"通过code 换取 access失败：{e}")
            return redirect(f"{param.url}",status=403)   
       
        return redirect(param.url )
        return method(*args, **kwargs)
        '''
        authorize_url:'https://open.weixin.qq.com/connect/oauth2/authorize?appid=123456&redirect_uri=http://localhost'
            '&response_type=code&scope=snsapi_base#wechat_redirect'
        authorize_url:'https://open.weixin.qq.com/connect/oauth2/authorize?appid=123456&redirect_uri=/index.html
                    &response_type=code&scope=snsapi_userinfo&state=stateCode#wechat_redirect
        qrconnect_url: 'https://open.weixin.qq.com/connect/qrconnect?appid=123456&redirect_uri=http://localhost'
            '&response_type=code&scope=snsapi_login#wechat_redirect'
        '''  
    return warpper

def oauth_debug(method):
    @wraps(method)
    async def  warpper(request:Request,param:Authon_param ):    
        uuid=await getAccountuuid(request,userOpenId="oPwvL6J2X9Ynytuo5agMLgKGVJQI") 
        res=redirect(f"{param.url}?ticket={uuid}")  
        return res 
    return warpper


async def get_snsapi_userinfo(oauth:WeChatOAuth,request:Request,param:Authon_param):
    """
    获取用户信息入库
    //todo 需要优化
    """
    try:
        if param.scope==wx_authon_scope.snsapi_userinfo: 
            # 第四步 获取用户信息
            user=oauth.get_user_info()  
            wx_user=wxUser()
            wx_user.__dict__.update(user)
            wx_user.ownedAppid=param.appid
            log.warn(f"数据库中privilege特权存储NULL.{ wx_user.privilege}，如有需要请调整代码")
            wx_user.privilege="" 
           
            #asyncio.run(oauth_wx_user_to_db(request,wxUser)) #已经启动的事件循环中调用 asyncio.run()，就会出现事件循环冲突的问题
            #loop = asyncio.get_event_loop()
            #loop.run_until_complete(oauth_wx_user_to_db(request,wxUser)) 
            await oauth_wx_user_to_db(request,wx_user) 
            uuid=await getAccountuuid(request,userOpenId=wx_user.openid) 
            log.err(f"{param.url}?ticket={uuid}")
            return redirect(f"{param.url}?ticket={uuid}") 
            #,headers={"Authorization":token}
            #res.add_cookie("Authorization",token) 
            #return res
        else:
            log.warn(param.appid) 
    except Exception as e:
        log.warn(f"获取微信用户失败：{e}")
        return redirect(f"{param.url}",status=403) 