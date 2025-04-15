from sanic import Sanic, Blueprint, Request
from api.sys.syscode import api_dynamic, api_code
from api.sys.sysTask import task_api

sys_api = Blueprint.group(api_dynamic, api_code, task_api,  url_prefix="/sys")
