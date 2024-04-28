from sanic import Sanic, Blueprint,Request

from ..view_model.user_group import user_groups_view,user_group_view,user_groups_tree_view

userGroup_api = Blueprint("user_group_API",url_prefix="/userGroup") 

userGroup_api.add_route(user_groups_view.as_view(),"/",name=user_groups_view.__name__) 
userGroup_api.add_route(user_groups_tree_view.as_view(),"/tree",name=user_groups_tree_view.__name__) 
userGroup_api.add_route(user_group_view.as_view(),"/<pk:int>",name=user_group_view.__name__) 