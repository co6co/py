from api.view_model.wx import wx_base_view
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
from model import WechatConfig


class Wx_message_View(wx_base_view):
    """
    与页面相关，页面中不能出现 与微信 openid 相关的东西
    """
    def get(self):
        self.client.media.upload()


class WxView_Api(wx_base_view):
    """
    群发消息
    主动设置 clientmsgid 来避免重复推送
    群发保护-->需要等待管理员进行确认
    1. 将图文消息中需要用到的图片，使用上传图文图片接口，上传成功并获得图片 URL
    2. 用户标签的群发，或对 OpenID 列表的群发，将图文消息群发出去，群发时微信会进行原创校验，并返回群发操作结果；
    3. 如果需要，还可以预览图文消息、查询群发状态，或删除已群发的消息等

    素材管理接口-->mediaID

    is_to_all=true--->使其进入公众号在微信客户端的历史消息列表【media_id 会失效，后台草稿也会被自动删除。】
    """
    async def post(self, request:Request):
        """

        """ 
        self.client.message.send_mass_article()
        return text("")


