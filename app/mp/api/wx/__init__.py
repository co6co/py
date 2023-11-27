from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from api.view_model.wx.wx_view  import WxView
from api.view_model.wx.media_view  import Media_View

wx_api = Blueprint("wx_api")
# 微信服务器请求URL
wx_api.add_route(WxView.as_view(),"/wx/<appid:str>",name="wx")

