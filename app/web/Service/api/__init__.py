from sanic import Blueprint, Websocket

from sanic.response import text
from sanic.exceptions import NotFound

from co6co_permissions.api import permissions_api
from co6co_task.api import tasks_api
from api.biz import biz_api
from api.tools import tool_api
from api.ai import Api as ai_api
from api.transmit import transmit_api
from api.dev import dev_api
all_api = [permissions_api, tasks_api, biz_api, tool_api, ai_api, transmit_api, dev_api]
api = Blueprint.group(*all_api, url_prefix="/api", version=1)
