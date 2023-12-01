from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.resource_view import Resources_View,Resource_View


biz_api = Blueprint("resource_API")
# 与盒子对接，请求，不需要认证
biz_api.add_route(Resources_View.as_view(),"/resource",name="resources")
biz_api.add_route(Resource_View.as_view(),"/resource/<uid:str>",name="resource") 