from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.wx.wx_view  import WxView
from view_model.wx.media_view  import Media_View
from view_model.wx.api_view  import config_View, menus_Api,menu_Api
from view_model.wx.page_view  import Authon_View,Authon_debug_View
from view_model.wx.api_template_view import template_message_View

wx_api = Blueprint("wx_API")
# 微信服务器请求URL
wx_api.add_route(WxView.as_view(),"/wx/<appid:str>",name="wx")
wx_api.add_route(config_View.as_view(),"/wx/config",name="配置")

wx_api.add_route(menus_Api.as_view(),"/wx/menu",name="公众号菜单s")
wx_api.add_route(menu_Api.as_view(),"/wx/menu/<pk:int>",name="公众号菜单")

wx_api.add_route(Authon_View.as_view(),"/wx/<appid:str>/oauth2",name="微信snsapi ") #snsapi_base/snsapi_userinfo 
wx_api.add_route(Authon_debug_View.as_view(),"/wx/<appid:str>/oauth_debug",name="微信snsapi2 ") #snsapi_base/snsapi_userinfo 
# 模板信息
wx_api.add_route(template_message_View.as_view(),"/wx/template",name=template_message_View.__name__) #snsapi_base/snsapi_userinfo 





