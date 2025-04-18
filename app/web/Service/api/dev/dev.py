from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes
from view_model.device import View, Views, ExistView, ImportView, DeviceCategoryView

_dev_api = Blueprint("dev_API")
add_routes(_dev_api, ExistView, View, Views, ImportView, DeviceCategoryView)
