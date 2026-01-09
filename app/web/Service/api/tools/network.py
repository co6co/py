from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.tools.network import View
network_api = Blueprint("network", url_prefix="/network")
add_routes(network_api,   View)
