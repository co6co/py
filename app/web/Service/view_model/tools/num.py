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
from co6co.utils import log


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
    allCategoryList = [0, 1, 2]

    async def get(self, request: Request, category: int):
        if category == -1:
            return self.response_json(Result.success(data.category.to_dict_list()))
        if category not in self.allCategoryList:
            return self.response_json(Result.fail(self.allCategoryList, "所给类别不在预置类别中"))

        category: data.category = data.category.val2enum(category)
        return self.response_json(Result.success(category.toDesc()))

    async def post(self, request: Request, category: int):
        json: dict = request.json
        lst = json.get("list")
        danList: list = json.get("dans", [])
        category: data.category = data.category.val2enum(category)
        rest = category.padding(lst,  *danList)
        return self.response_json(Result.success(rest))
