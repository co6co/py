from sanic import Sanic, Blueprint, Request
from api.dev.dev import _dev_api
from api.dev.image import _img_api

dev_api = Blueprint.group(_dev_api, _img_api, url_prefix="/dev")
