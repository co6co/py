from ..api.menu import menu_api
from ..api.user import user_api
from .aop.api_auth import authorized
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result

from ..model.enum import menu_type, menu_state,user_state

"""
__init__.py 还是有其他用处的

菜单
""" 
@menu_api.route("/status", methods=["GET", "POST"])
@authorized
async def getMenuStatus(request: Request):
    """
    菜单状态
    """
    states = menu_state.to_dict_list()
    return JSON_util.response(Result.success(data=states))


@menu_api.route("/category", methods=["GET", "POST"])
@authorized
async def getMenuCategory(request: Request):
    """
    菜单类别
    """
    states = menu_type.to_dict_list()
    return JSON_util.response(Result.success(data=states))

"""
用户
"""
user_api.route("/status", methods=["GET", "POST"])
@authorized
async def getUserStatus(request: Request):
    """
    用户状态
    """
    states = user_state.to_dict_list()
    return JSON_util.response(Result.success(data=states))


