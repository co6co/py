from sanic import Blueprint
from co6co_sanic_ext.api import add_routes
from view_model.deep import DeepseekView, TmView, SgView

Api = Blueprint("ai_api", url_prefix="/ai")
add_routes(Api, DeepseekView, TmView, SgView)
