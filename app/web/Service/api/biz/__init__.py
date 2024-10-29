from sanic import Sanic, Blueprint, Request
from api.biz.sysTask import api as sys_task_api
from api.biz.ui import api as components_api
biz_api = Blueprint.group(sys_task_api, components_api, url_prefix="/biz")
