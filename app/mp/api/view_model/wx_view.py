from co6co_sanic_ext.view_model import BaseMethodView,Request 
from sanic.response import text,raw
from typing import List,Optional

# wechatpy 依赖 cryptography
# http://docs.wechatpy.org/zh-cn/stable/quickstart.html
from wechatpy.crypto import WeChatCrypto
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.exceptions import InvalidAppIdException
from co6co.utils import log
from utils import wx_message
from model import WechatConfig

class WxView(BaseMethodView):
    def _get_config(self,request:Request,appid:str)->Optional[ WechatConfig]:
        configs:List[dict]=request.app.config.wx_config  
        filtered:filter= filter(lambda c:c.get("appid")==appid,configs)
        
        config=WechatConfig()
        for f in filtered:
            config.__dict__.update(f)  
        return config
    def get(self,request:Request,appid:str):
        try:
            signature = request.args.get("signature")
            log.warn(f"signature:{signature}")
            config=self._get_config(request,appid) 
            if config and signature: 
                timestamp = request.args.get("timestamp")
                nonce = request.args.get("nonce")
                echostr = request.args.get("echostr")
                encrypt_type = request.args.get("encrypt_type", "") 
                msg_signature = request.args.get("msg_signature", "")
                check_signature(config.token, signature, timestamp, nonce)
                return text(echostr)
            return text(u"微信验证失败",403) 
        except Exception as e: 
            return text(f"异常请求{e}",403)
    @staticmethod
    @wx_message
    def message(request:Request,msg:any,config:WechatConfig):
        if msg.type == "text":
            reply = create_reply(f"回复消息，我收到你的信息了：{msg.content}",msg )
        else:
            reply = create_reply("Sorry, can not handle this for now", msg)
        return reply

    def post(self,request:Request,appid:str): 
        try: 
            header=dict({"Content-Type":"text/xml"}) 
            config:WechatConfig=self._get_config(request,appid)
            timestamp = request.args.get("timestamp")
            nonce = request.args.get("nonce")
            msg_signature = request.args.get("msg_signature", "")
            crypto = WeChatCrypto(config.token,config.encodingAESKey, appid)
            try:
                msg = crypto.decrypt_message(request.body, msg_signature, timestamp, nonce)
                print(f"from:{appid} Decrypted message: \n{msg}")
            except (InvalidSignatureException, InvalidAppIdException) as e: 
                reply = create_reply("出错", e)
                return raw(crypto.encrypt_message(reply.render(), nonce, timestamp),403,headers=header)
            msg = parse_message(msg) 
            reply=WxView.message(request,msg,config) 
            return raw(crypto.encrypt_message(reply.render(), nonce, timestamp),headers=header)
        except Exception as e:
            log.warn(e)

        