from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.device import View, Views, ExistView, ImportView
from view_model.device.imgView import Views,  PreView

_img_api = Blueprint("img_API", url_prefix="/img")
add_routes(_img_api, Views,  PreView)
