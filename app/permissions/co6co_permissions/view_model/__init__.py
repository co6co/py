from ..api.menu import menu_api
from .aop.api_auth import authorized
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from .base_view import AuthMethodView
from ..model.pos.right import menuPO
from ..model.filters.menu_filter import menu_filter
from ..model.enum import menu_type, menu_state


@menu_api.route("/status", methods=["GET", "POST"])
@authorized
async def getMenuStatus(request: Request):
    states = menu_state.to_dict_list()
    return JSON_util.response(Result.success(data=states))


@menu_api.route("/category", methods=["GET", "POST"])
@authorized
async def getMenuCategory(request: Request):
    states = menu_type.to_dict_list()
    return JSON_util.response(Result.success(data=states))
