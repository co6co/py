from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.site_view import Site_View,Sites_View ,Site_config_View

site_api = Blueprint("site_api")
 
site_api.add_route(Site_View.as_view(),"/biz/site/<pk:int>",name=Site_View.__name__)
site_api.add_route(Sites_View.as_view(),"/biz/site",name=Sites_View.__name__) 
site_api.add_route(Site_config_View.as_view(),"/biz/site/config/<pk:int>",name=Site_config_View.__name__) 