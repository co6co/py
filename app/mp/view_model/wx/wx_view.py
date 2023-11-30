from view_model.base_view import  AuthMethodView
from view_model.wx import wx_base_view
from sanic.response import text,raw
from typing import List,Optional
from sanic import  Request

# wechatpy 依赖 cryptography
# http://docs.wechatpy.org/zh-cn/stable/quickstart.html
from wechatpy.crypto import WeChatCrypto
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.exceptions import InvalidAppIdException
from co6co.utils import log
from utils import wx_message
from model.enum.wx import wx_encrypt_mode
from model import WechatConfig

class WxView(wx_base_view):
    """
    微信服务器 入口
    """

    def get(self,request:Request,appid:str):
        try:
            signature = request.args.get("signature") 
            config=self.get_wx_config(request,appid) 
            if config and signature: 
                timestamp = request.args.get("timestamp")
                nonce = request.args.get("nonce")
                echostr = request.args.get("echostr")
                encrypt_type = request.args.get("encrypt_type", "") 
                msg_signature = request.args.get("msg_signature", "")
                check_signature(config.token, signature, timestamp, nonce)
                return text(echostr)
            return text(u"验证失败",403) 
        except Exception as e: 
            return text(f"异常请求{e}",403)
    @staticmethod
    @wx_message
    async def message(request:Request,msg:any,config:WechatConfig):
        if msg.type == "text":
            reply = create_reply(f"回复消息，我收到你的信息了：{msg.content}",msg )
        else:
            reply = create_reply("Sorry, can not handle this for now", msg)
        return reply
    @staticmethod
    def create_WeChatCrypto(config:WechatConfig)->WeChatCrypto|None:
         if config.encrypt_mode==wx_encrypt_mode.safe.name:
            return WeChatCrypto(config.token,config.encodingAESKey, config.appid) 
         
    @staticmethod
    def getMessage(body:str,crypto:WeChatCrypto,msg_signature:str,timestamp:str,nonce:str):
        if crypto==None:return body
        return crypto.decrypt_message(body, msg_signature, timestamp, nonce)
    @staticmethod
    def reply_message(reply,crypto:WeChatCrypto,timestamp:str,nonce:str,status:int=200): 
        header=dict({"Content-Type":"text/xml"}) 
        if crypto!=None:
            return raw(crypto.encrypt_message(reply.render(), nonce, timestamp),status,headers=header)
        else: return raw(reply.render() ,status,headers=header)
        

    async def post(self,request:Request,appid:str): 
        try:  
            config:WechatConfig=self.get_wx_config(request,appid)
            timestamp = request.args.get("timestamp")
            nonce = request.args.get("nonce")
            msg_signature = request.args.get("msg_signature", "")
            crypto=WxView.create_WeChatCrypto(config)
            try: 
                msg=WxView.getMessage(request.body,crypto,msg_signature,timestamp, nonce) 
                print(f"from:{appid} Decrypted message: \n{msg}")
            except (InvalidSignatureException, InvalidAppIdException) as e: 
                reply = create_reply("出错", e) 
                return WxView.reply_message(reply,crypto, nonce, timestamp,403) 
            msg = parse_message(msg)  
            reply=await WxView.message(request,msg,config) 
            if reply ==None: # 来不及处理回复空串
                reply = create_reply(None)
                return WxView.reply_message(reply,crypto, nonce, timestamp) 
            return WxView.reply_message(reply,crypto, nonce, timestamp) 
        except Exception as e:
            log.warn(e)
            raise