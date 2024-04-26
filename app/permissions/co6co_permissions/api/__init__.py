from sanic import Sanic, Blueprint,Request

from ..view_model.user_group import user_groups_view,user_group_view,user_groups_tree_view

permissions_api = Blueprint("permissions_API",url_prefix="/permissions") 

permissions_api.add_route(user_groups_tree_view.as_view(),"/userGroup/tree",name=user_groups_tree_view.__name__) 
permissions_api.add_route(user_groups_view.as_view(),"/userGroup",name=user_groups_view.__name__) 
permissions_api.add_route(user_group_view.as_view(),"/userGroup/<pk:int>",name=user_group_view.__name__) 
