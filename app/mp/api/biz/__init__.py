from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file

# 上传告警
from view_model.biz.upload_view import Video_Upload_View,Alarm_Upload_View 

from view_model.biz.alarm_view import Alarm_category_View,Alarms_View,Alarm_View ,Alarm_uuid_View
from view_model.biz.device_view import Devices_View, IP_Cameras_View,IP_Camera_View,IP_Camera_poster_View


biz_api = Blueprint("biz_API")
# 与盒子对接，请求，不需要认证
biz_api.add_route(Video_Upload_View.as_view(),"/biz/upload/video",name=Video_Upload_View.__name__)
biz_api.add_route(Alarm_Upload_View.as_view(),"/biz/upload/alarm",name=Alarm_Upload_View.__name__)  

biz_api.add_route(Alarms_View.as_view(),"/biz/alarm",name=Alarms_View.__name__) 
biz_api.add_route(Alarm_View.as_view(),"/biz/alarm/<pk:int>",name=Alarm_View.__name__)  
biz_api.add_route(Alarm_uuid_View.as_view(),"/biz/alarm/<uuid:str>",name=Alarm_uuid_View.__name__)  

biz_api.add_route(Alarm_category_View.as_view(),"/biz/alarm/category",name=Alarm_category_View.__name__) 
biz_api.add_route(Devices_View.as_view(),"/biz/device",name=Devices_View.__name__) 
biz_api.add_route(IP_Cameras_View.as_view(),"/biz/device/camera",name=IP_Cameras_View.__name__) 
biz_api.add_route(IP_Camera_View.as_view(),"/biz/device/camera/<pk:int>",name=IP_Camera_View.__name__) 
biz_api.add_route(IP_Camera_poster_View.as_view(),"/biz/device/poster/<pk:int>",name=IP_Camera_poster_View.__name__)  