from sanic import Blueprint
from .wx import wx_api 
from sanic.response import  text
from sanic.exceptions import NotFound
from api.wx import wx_api

api = Blueprint.group(wx_api , url_prefix="/api",version=1)