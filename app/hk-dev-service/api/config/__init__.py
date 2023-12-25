from sanic import Sanic, Blueprint, Request
from sanic.response import json, file_stream, file

from view_model.config_view import Configs_View as views, Config_View as view
config_api = Blueprint("Config_API")


config_api.add_route(views.as_view(), "/sys/config", name="configs")
config_api.add_route(view.as_view(), "/sys/config/<pk:int>", name="config")
