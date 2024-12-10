from sanic import Sanic, Blueprint, Request
from api.tools.num import num_api
tool_api = Blueprint.group(num_api, url_prefix="/tools")
