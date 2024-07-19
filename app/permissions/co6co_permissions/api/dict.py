
from sanic import Sanic, Blueprint, Request
from ..view_model.dict.dictTypeView import DictTypeViews, DictTypeView, DictTypeExistView
from ..view_model.dict.dictView import Views, View
from co6co_sanic_ext.api import add_routes

dict_api = Blueprint("dict_API", url_prefix="/dict")
add_routes(dict_api, Views, View, )
add_routes(dict_api,  DictTypeViews, DictTypeView, DictTypeExistView)
