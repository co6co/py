from sanic import Blueprint
 
from sanic.response import  text
from sanic.exceptions import NotFound
from api.user import user_api
from api.wx import wx_api
from api.biz import  biz_api
from api.biz.server import  server_api
from api.biz.site import  site_api
from api.biz.resource import  resource_api
from api.test import test_api 



api = Blueprint.group(user_api,wx_api ,site_api,biz_api,resource_api,server_api,test_api, url_prefix="/api",version=1)