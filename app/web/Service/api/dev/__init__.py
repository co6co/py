from sanic import Sanic, Blueprint, Request
from api.dev.dev import _dev_api

dev_api = Blueprint.group(_dev_api,  url_prefix="/dev")
