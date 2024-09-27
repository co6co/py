from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.systask import View, Views, ExistView
api = Blueprint("sysTask_API", url_prefix="/sys/task")
add_routes(api, ExistView, View, Views)
