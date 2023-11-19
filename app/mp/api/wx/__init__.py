from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from api.view_model.wx_view  import WxView

wx_api = Blueprint("wx_api" )
wx_api.add_route(WxView.as_view(),"/wx/<appid:str>",name="wx")  