
from functools import wraps
from co6co.utils import log
from wechatpy import WeChatClient
from sanic.request import Request
from sanic.response import redirect,raw
from model import WechatConfig
from model.enum.wx import wx_message_type
from wechatpy import messages ,events

from wechatpy.oauth import WeChatOAuth
from utils.db import wx_open_id_into_db


def oauth(func):
    def warpper(request:Request):
        if request.session.get('user_info', None) is None:
            code = request.GET.get('code', None)
            wechat_oauth = getWeChatOAuth(request.get_raw_uri())
            url = wechat_oauth.authorize_url
            if code:
                try:
                    wechat_oauth.fetch_access_token(code)
                    user_info = wechat_oauth.get_user_info()
                except Exception as e:
                    print(str(e))
                    # 这里需要处理请求里包含的 code 无效的情况
                    # abort(403)
                else:
                    request.session['user_info'] = user_info
            else:
                return redirect(url)
 
        return func(request)
    return warpper

@oauth
def get_wx_user_info(request:Request): 
    user_info = request.session.get('user_info')
    return raw(str(user_info))


def remove_repetition_message(func):
    message_ids=[]
    remove_message_ids=[]
    @wraps(func)
    def remove(*args,**kwargs): 
        log.start_mark("-----------")
        msgId=None
        try:
            for i  in args: 
                if isinstance(i,messages.BaseMessage): 
                    msg:messages.BaseMessage=i
                    if msg.id in message_ids: 
                        #log.warn("过滤掉重复消息！")
                        return
                    message_ids.append(msg.id)
                    msgId=msg.id  
            return func(*args, **kwargs)
        finally:
            #log.log("123456")
            #log.end_mark("------end-----")
            if msgId!=None: remove_message_ids.append(msgId)
            #else:log.warn("has : ‘过滤掉重复消息！’ message")
    return remove

'''
群发消息, 被动消息，客服消息，模板消息
公众号内网页：复杂的业务场景，需要通过网页形式来提供服务
    网页授权获取用户基本信息：通过该接口，可以获取用户的基本信息（获取用户的OpenID是无需用户同意的，获取用户的基本信息则需用户同意）
    微信JS-SDK，JavaScript代码使用微信原生功能的工具包，录制和播放微信语音、监听微信分享、上传手机本地图片、拍照等许多能力
'''     
def wx_message(func):
    """
    对被动回复消息进行处理
    """ 
    @wraps(func)
    #@remove_repetition_message
    async def check_message(request:Request,msg:messages.BaseMessage ,config:WechatConfig, *args, **kwargs):
        # 消息收到最多三次 注意去重 
        log.info(f"消息类型：{type(msg)}{msg.type},{msg}")
        if wx_message_type.text.getName()==msg.type:
            return await _wx_text(request,msg,config) 
        if wx_message_type.image.getName()==msg.type:
            return await _wx_image(request,msg,config)
        if wx_message_type.voice.getName()==msg.type:
            return await _wx_voice(request,msg,config)
        if wx_message_type.video.getName()==msg.type:
            return await _wx_video(request,msg,config)
        if wx_message_type.link.getName()==msg.type:
            return await _wx_link(request,msg,config)
        if wx_message_type.location.getName()==msg.type:
            return await _wx_location(request,msg,config)
        if wx_message_type.event.getName()==msg.type:
            return await _wx_event(request,msg,config) 
        
        return await func(request,msg,config,*args, **kwargs)
    return check_message 

async def _wx_text(request:Request,msg:messages.TextMessage,config:WechatConfig): 
    #TextMessage({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn632hnZYU9v1FcO26trhghW4', 'CreateTime': '1700727977', 'MsgType': 'text', 'Content': '文本', 'MsgId': '24348569934556997'})
    log.warn(f"文本消息：{msg.type},{msg}") 
    getUser(config,msg) #Error code: 48001, message: api unauthorized rid: 65640ec8-74a2c5fc-55771ced
async def _wx_image(request:Request,msg:messages.ImageMessage,config:WechatConfig):
     #ImageMessage({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn632hnZYU9v1FcO26trhghW4', 'CreateTime': '1700728033', 'MsgType': 'image', 'PicUrl': 'http://mmbiz.qpic.cn/sz_mmbiz_jpg/icrw9KdJuAHNbBoDicMHqcG8ftkh0S6yeqxPg9UML9ZAq34hnPWqWRqicWxVGXrhq4zlF7haXD4cYTt0o5IyEGKeQ/0', 'MsgId': '24348570211378290', 'MediaId': 'LYgeZgfZ7t_t6N7idjUEz3-9VElNVeyvhOkAacIJe71hcImo2j12teMt3KyP8buO'})
     log.warn(f"image消息：{msg.type},{msg}")
async def _wx_voice(request:Request,msg:messages.VoiceMessage,config:WechatConfig):
     #VoiceMessage({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn632hnZYU9v1FcO26trhghW4', 'CreateTime': '1700728068', 'MsgType': 'voice', 'MediaId': 'LYgeZgfZ7t_t6N7idjUEz7O9sdQT8jN0yHNM20XX4sYhqJbfxHycDJ0yTE1ZfeQs', 'Format': 'amr', 'MsgId': '24348573055098068', 'Recognition': None})
     log.warn(f"voice消息：{msg.type},{msg}")
async def _wx_video(request:Request,msg:messages.VoiceMessage,config:WechatConfig):
     log.warn(f"video消息：{msg.type},{msg}")
async def _wx_link(request:Request,msg:messages.LinkMessage,config:WechatConfig):
     log.warn(f"link消息：{msg.type},{msg}")
async def _wx_location(request:Request,msg:messages.LocationMessage,config:WechatConfig):
     #location,LocationMessage({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn632hnZYU9v1FcO26trhghW4', 'CreateTime': '1700728162', 'MsgType': 'location', 'Location_X': '25.657160', 'Location_Y': '103.558861', 'Scale': '0', 'Label': '大坡乡', 'MsgId': '24348572679095274'})
     log.warn(f"location消息：{msg.type},{msg}")

@wx_open_id_into_db
async def _wx_event(request:Request,msg:events.BaseEvent,config:WechatConfig):
    tt=events.SubscribeEvent(msg) 
    getUser(config, tt)
     #SubscribeEvent({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn61ohODXRgz4Z-u4GIYVBez0', 'CreateTime': '1700728265', 'MsgType': 'event', 'Event': 'subscribe', 'EventKey': None})
     #UnsubscribeEvent({'ToUserName': 'gh_c8b421a2ed81', 'FromUserName': 'otcIn61ohODXRgz4Z-u4GIYVBez0', 'CreateTime': '1700728306', 'MsgType': 'event', 'Event': 'unsubscribe', 'EventKey': None})
    log.warn(f"事件消息：{msg.type},{msg}")


def getUser(config:WechatConfig,msg:messages.BaseMessage):
    log.succ(f"config:{config.appid}{ config.appSecret}")
    wxClient =WeChatClient(config.appid, config.appSecret) 
    log.succ(f"{wxClient.access_token_key}:{wxClient.access_token},{wxClient.expires_at}")  
    print( "菜单：",wxClient.menu.get())
    log.err(f"用户ID：{msg.source}")
    wxUserInfo = wxClient.user.get(msg.source)
    log.err(f"用户信息：{wxUserInfo}")
    
def getWeChatOAuth(redirect_url,config:WechatConfig):
    return WeChatOAuth(config.appid, config.appSecret, redirect_url)