from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.components import componentViews
api = Blueprint("custom_ui_API", url_prefix="/components")
add_routes(api, componentViews)
