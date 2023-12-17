from sanic import Blueprint
 
from sanic.response import  text
from sanic.exceptions import NotFound
from api.user import user_api 
from api.device import device_api 
from api.user.user_task import  task_api

api = Blueprint.group(user_api,device_api ,task_api,url_prefix="/api",version=1)