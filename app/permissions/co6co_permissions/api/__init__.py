from sanic import Sanic, Blueprint,Request

from ..view_model.user_group import user_groups_view

permissions_api = Blueprint("permissions_API",url_prefix="/permissions") 

permissions_api.add_route(user_groups_view.as_view(),"/userGroup",name=user_groups_view.__name__) 
