from sanic import Blueprint

from sanic.response import text
from sanic.exceptions import NotFound
from api.user import user_api
from api.device import device_api
from api.user.user_task import task_api
from api.config import config_api

api = Blueprint.group(config_api,user_api, device_api, task_api, url_prefix="/api", version=1)
