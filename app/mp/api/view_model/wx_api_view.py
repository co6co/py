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


class Wx_message_View(BaseMethodView):
    """
    与页面相关，页面中不能出现 与微信 openid 相关的东西
    """


