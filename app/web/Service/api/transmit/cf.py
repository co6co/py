from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.transimt.cf import ListView, OneView
_api = Blueprint("cf", url_prefix="/cf")
add_routes(_api, ListView, OneView)
