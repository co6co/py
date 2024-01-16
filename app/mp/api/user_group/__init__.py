from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.user_group.group_view import Group_View,Groups_View

group_api = Blueprint("group_api")

group_api.add_route(Group_View.as_view(),"/user/group/<pk:int>",name=Group_View.__name__)
group_api.add_route(Groups_View.as_view(),"/user/group",name=Groups_View.__name__) 