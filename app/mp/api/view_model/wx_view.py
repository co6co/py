from co6co_sanic_ext.view_model import BaseMethodView,Request
from sanic.response import text 
from typing import List,Optional

from wechatpy.crypto import WeChatCrypto
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.exceptions import InvalidAppIdException
from co6co.utils import log
 
class WechatConfig:
    """
    配置信息[开发信息| 服务器配置]
    开发信息:appid,appSecret
    服务器配置:url, token,encodingAESKey,encrypt_mode
    
    """
    appid:str=None,		                #    公众号的appid
    appSecret:str=None,
    
    token:str=None, 		            #    token
    encodingAESKey:str=None,		            # 公众号的secret
    encrypt_mode:str=None
    def __init__(self) -> None:
        pass

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
            raise e
            return text(f"异常请求{e}",404)
    def post(self,request:Request,appid:str):
        print(f"Raw message: \n{request.data}")
        config:WechatConfig=self._get_config(request,appid)
        timestamp = request.args.get("timestamp")
        nonce = request.args.get("nonce")
        msg_signature = request.args.get("msg_signature", "")
        crypto = WeChatCrypto(config.token,config.encodingAESKey, appid)
        try:
            msg = crypto.decrypt_message(request.data, msg_signature, timestamp, nonce)
            print(f"Decrypted message: \n{msg}")
        except (InvalidSignatureException, InvalidAppIdException):
            return text("",status=403)
        msg = parse_message(msg)
        if msg.type == "text":
            reply = create_reply(msg.content, msg)
        else:
            reply = create_reply("Sorry, can not handle this for now", msg)
        return crypto.encrypt_message(reply.render(), nonce, timestamp)

        