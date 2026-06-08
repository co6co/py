from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes,add_websocket_route 

_api = Blueprint("demo", url_prefix="/demo")
 