from sanic import Sanic, Blueprint, Request
from api.tools.num import num_api
from api.tools.network import network_api

tool_api = Blueprint.group(num_api, network_api, url_prefix="/tools")

