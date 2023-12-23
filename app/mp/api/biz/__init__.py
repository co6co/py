from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.upload_view import Video_Upload_View,Alarm_Upload_View
from view_model.biz.alarm_view import Alarms_View,Alarm_View
from view_model.biz.device_view import Devices_View, IP_Cameras_View,IP_Camera_View,IP_Camera_poster_View


biz_api = Blueprint("biz_API")
# 与盒子对接，请求，不需要认证
biz_api.add_route(Video_Upload_View.as_view(),"/biz/upload/video",name="upload_video")
biz_api.add_route(Alarm_Upload_View.as_view(),"/biz/upload/alarm",name="upload_alarm") 
biz_api.add_route(Alarms_View.as_view(),"/biz/alarm",name="alarms") 
biz_api.add_route(Alarm_View.as_view(),"/biz/alarm/<pk:int>",name="alarm") 

biz_api.add_route(Devices_View.as_view(),"/biz/device",name=Devices_View.__name__) 
biz_api.add_route(IP_Cameras_View.as_view(),"/biz/device/camera",name=IP_Cameras_View.__name__) 
biz_api.add_route(IP_Camera_View.as_view(),"/biz/device/camera/<pk:int>",name=IP_Camera_View.__name__) 
biz_api.add_route(IP_Camera_poster_View.as_view(),"/biz/device/poster/<pk:int>",name=IP_Camera_poster_View.__name__) 