from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import TaskPO
from view_model._filters.sysTask import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute
from view_model.tools import data
from model.enum import User_category


class Views(AuthMethodView):
    async def get(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(TaskPO.id, TaskPO.name, TaskPO.code, TaskPO.state, TaskPO.execStatus)
            .order_by(TaskPO.code.asc())
        )
        return await self.query_list(request, select,  isPO=False)

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        param.__dict__.update(request.json)

        return await self.query_page(request, param)


class View(AuthMethodView):
    routePath = "/<category:int>"

    async def post(self, request: Request, category: int):
        json: dict = request.json
        lst = json.get("list")
        danList: list = json.get("dans")
        rest = []
        if category == 0:
            rest = data.padding(lst, data.arr_10_7_6)
        elif category == 1:
            rest = data.padding(lst, data.arr_15_7_5)
        elif category == 2:
            rest = data.padding(lst, data.arr_10_1_7_6, *danList)

        return self.response_json(Result.success(rest))
