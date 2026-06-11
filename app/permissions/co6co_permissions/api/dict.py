
from sanic import  Blueprint
from co6co_sanic_ext.api import add_routes
from ..view_model.dict.dictTypeView import DictTypeViews, DictTypeView, DictTypeExistView
from ..view_model.dict.dictView import Views, View, DictSelectView


_dict_api = Blueprint("dict_API")
add_routes(_dict_api, Views, View, DictSelectView)
_dictType_api = Blueprint("dict_type_API",url_prefix="/type")
add_routes(_dictType_api,  DictTypeViews, DictTypeView, DictTypeExistView)
dictAll_api = Blueprint.group(_dict_api,_dictType_api,url_prefix="/dict")
