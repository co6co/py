from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.systask import View, Views, ExistView
from view_model.systask.schedView import schedView, cronViews, codeView
api = Blueprint("sysTask_API", url_prefix="/sys/task")
add_routes(api, ExistView, View, Views)
add_routes(api, schedView, cronViews, codeView)
