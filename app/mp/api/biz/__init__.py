from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.upload_view import Video_Upload_View,Alarm_Upload_View
from view_model.biz.alarm_view import Alarms_View,Alarm_View


biz_api = Blueprint("biz_API")
# 与盒子对接，请求，不需要认证
biz_api.add_route(Video_Upload_View.as_view(),"/biz/upload/video",name="upload_video")
biz_api.add_route(Alarm_Upload_View.as_view(),"/biz/upload/alarm",name="upload_alarm") 
biz_api.add_route(Alarms_View.as_view(),"/biz/alarm",name="alarms") 
biz_api.add_route(Alarm_View.as_view(),"/biz/alarm/<pk:int>",name="alarm") 