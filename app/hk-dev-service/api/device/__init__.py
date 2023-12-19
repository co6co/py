from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
  
from view_model import device_view 
device_api = Blueprint("device_API")


device_api.add_route(device_view.Device_Category_View.as_view(),"/biz/category",name="category") 
device_api.add_route(device_view.Device_View.as_view(),"/biz/device",name="devices") 
