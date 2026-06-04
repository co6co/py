from sanic import  Blueprint, Request
from co6co_sanic_ext .api import add_routes
from ..view_model.user import user_ass_view, users_view, user_view, sys_users_view, ticketView,user_query_view,user_exist_view,user_exist_post_view
from ..view_model.currentUser import changePwd_view, user_info_view, user_avatar_view
from ..view_model.login import login_view
from ..view_model.aop.api_auth import authorized
from ..model.enum import user_state, user_category 
from co6co.data.result import Result
from co6co_sanic_ext.view_model import response_json
 
import random
import string


user_api = Blueprint("users_API", url_prefix="/users")
add_routes(user_api, login_view, ticketView)
add_routes(user_api, changePwd_view, user_info_view, user_avatar_view)
add_routes(user_api, user_ass_view, users_view, user_view, sys_users_view,user_query_view,user_exist_view,user_exist_post_view)


@user_api.route("/status", methods=["GET", "POST"])
@authorized
async def getUserStatus(request: Request):
    """
    用户状态
    """
    states = user_state.to_dict_list()
    return response_json(Result.success(data=states))


@user_api.route("/category", methods=["GET", "POST"])
@authorized
async def getUserCategory(request: Request):
    """
    用户类别
    """
    states = user_category.to_dict_list()
    return response_json(Result.success(data=states)) 
 


@user_api.route("/generatePwd/<length:int>", methods=["GET", "POST"])
async def generatePwd2(request: Request, length: int = 256):
    """
    生成随机密码 /generatePwd/<length:int>

    """
    all_characters = string.ascii_letters + string.digits  # + string.punctuation
    password = ''.join(random.choice(all_characters) for _ in range(length))
    return response_json(Result.success(data=password))


@user_api.route("/generatePwd", methods=["GET", "POST"])
async def generatePwd(request: Request):
    """
    生成随机密码 /generatePwd/256

    """
    return await generatePwd2(request, 256)
