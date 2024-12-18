from sanic import Blueprint, Websocket

from sanic.response import text
from sanic.exceptions import NotFound

from co6co_permissions.api import permissions_api
from api.biz import biz_api
from api.tools import tool_api

api = Blueprint.group(permissions_api, biz_api, tool_api, url_prefix="/api", version=1)
