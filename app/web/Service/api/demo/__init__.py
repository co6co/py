from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes,add_websocket_route 
from view_model import demoView,demoView2
_api = Blueprint("demo", url_prefix="/demo")
add_routes(_api, demoView,demoView2) 