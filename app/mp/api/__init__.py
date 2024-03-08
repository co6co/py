from sanic import Blueprint
 
from sanic.response import  text
from sanic.exceptions import NotFound

from api.app import app_api
from api.user import user_api
from api.user_group import group_api
from api.wx import wx_api
from api.biz import  biz_api
from api.biz.server import  server_api
from api.biz.site import  site_api
from api.biz.resource import  resource_api
from api.test import test_api 
from api.xss import xss_api 
from api.xss.topic import top_api
 
api = Blueprint.group(app_api,user_api,group_api,wx_api ,site_api,biz_api,resource_api,server_api,test_api,xss_api,top_api, url_prefix="/api",version=1)