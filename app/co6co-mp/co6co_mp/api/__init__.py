from sanic import Blueprint

from sanic.response import text
from sanic.exceptions import NotFound


from api.wx import wx_api
from co6co_permissions.api import permissions_api

api = Blueprint.group(permissions_api, wx_api, url_prefix="/api", version=1)
