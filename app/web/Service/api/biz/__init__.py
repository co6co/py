from sanic import Blueprint
from api.biz.ui import api as components_api
biz_api = Blueprint.group(components_api, url_prefix="/biz")
