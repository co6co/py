from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from api.view_model.wx.wx_view  import WxView
from api.view_model.wx.media_view  import Media_View
from api.view_model.wx.api_view  import config_View, menus_Api,menu_Api

wx_api = Blueprint("wx_API")
# 微信服务器请求URL
wx_api.add_route(WxView.as_view(),"/wx/<appid:str>",name="wx")
wx_api.add_route(config_View.as_view(),"/wx/config",name="配置")

wx_api.add_route(menus_Api.as_view(),"/wx/menu",name="公众号菜单s")
wx_api.add_route(menu_Api.as_view(),"/wx/menu/<pk:int>",name="公众号菜单")

