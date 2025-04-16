from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.tools.num import View
num_api = Blueprint("num", url_prefix="/num")
add_routes(num_api,   View)
