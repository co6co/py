from sanic import Sanic, Blueprint, Request
from api.transmit.cf import _api as cf_api
transmit_api = Blueprint.group(cf_api, url_prefix="/transmit")
