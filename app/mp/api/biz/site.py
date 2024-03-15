from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.site_view import Site_View,Sites_View ,Site_config_View
from view_model.biz.devices.box import Boxs_View,Box_View
from view_model.biz.devices.ipCaram import IpCameras_View,IpCamera_View
from view_model.biz.devices.router import Routers_View,Router_View

site_api = Blueprint("site_api")
# 站点
site_api.add_route(Site_View.as_view(),"/biz/site/<pk:int>",name=Site_View.__name__)
site_api.add_route(Sites_View.as_view(),"/biz/site",name=Sites_View.__name__) 
site_api.add_route(Site_config_View.as_view(),"/biz/site/config/<pk:int>",name=Site_config_View.__name__) 
# ai 盒子
site_api.add_route(Box_View.as_view(),"/biz/dev/aiBox/<pk:int>",name=Box_View.__name__)
site_api.add_route(Boxs_View.as_view(),"/biz/dev/aiBox",name=Boxs_View.__name__) 
# 监控球机
site_api.add_route(IpCamera_View.as_view(),"/biz/dev/ipCamera/<pk:int>",name=IpCamera_View.__name__)
site_api.add_route(IpCameras_View.as_view(),"/biz/dev/ipCamera",name=IpCameras_View.__name__) 
# 路由器
site_api.add_route(Router_View.as_view(),"/biz/dev/router/<pk:int>",name=Router_View.__name__)
site_api.add_route(Routers_View.as_view(),"/biz/dev/router",name=Routers_View.__name__) 

