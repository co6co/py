from sanic import Blueprint,json
from sanic.request import Request
from datetime import datetime
from co6co_web_db.utils import JSON_util
 
from api.debug import server_api
from api.signal import signal_api
api = Blueprint.group(server_api,signal_api, url_prefix="/rest" )
