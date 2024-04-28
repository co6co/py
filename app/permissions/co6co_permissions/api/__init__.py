from sanic import Sanic, Blueprint,Request
from .menu import menu_api
from .userGroup import userGroup_api

permissions_api=Blueprint.group(menu_api,userGroup_api,url_prefix="/permissions")
 

 