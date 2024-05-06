from sanic import Sanic, Blueprint,Request
from .menu import menu_api
from .userGroup import userGroup_api
from .role import role_api
from .user import user_api

permissions_api=Blueprint.group(menu_api,userGroup_api,role_api,user_api)
 

 