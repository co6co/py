from sanic import Sanic, Blueprint, Request
from co6co_sanic_ext.api import add_routes,add_websocket_route
from view_model.transimt.cf import ListView, OneView
from view_model.websocket.rtsp import client_vod
ws_api = Blueprint("ws", url_prefix="/ws")
add_routes(ws_api, ListView, OneView)

add_websocket_route(ws_api,client_vod.websocket_stream,routePath=client_vod.routePath)

