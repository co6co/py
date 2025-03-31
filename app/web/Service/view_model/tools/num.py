from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession
import re
from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools, QueryOneCallable
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import DynamicCodePO
from view_model._filters.sysTask import Filter
from view_model.tools import data
from model.enum import User_category
from co6co.utils import log
from co6co_permissions.model.pos.other import sysDictPO


class Views(AuthMethodView):
    async def get(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(DynamicCodePO.id, DynamicCodePO.name, DynamicCodePO.code, DynamicCodePO.state, DynamicCodePO.execStatus)
            .order_by(DynamicCodePO.code.asc())
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
    # 类别为字典Id
    routePath = "/<category:int>"

    async def get(self, request: Request, category: int):
        select = (
            Select(sysDictPO.flag, sysDictPO.id)
            .filter(sysDictPO.id.__eq__(category))
        )
        db = QueryOneCallable(self.get_db_session(request))
        dictOne: dict = await db(select=select, isPO=False)
        # dictOne: dict = await db_tools.execForMappings(self.get_db_session(request), select, True)
        value = dictOne.get("flag", "")
        return self.response_json(Result.success(data.toDesc(value)))

    async def post(self, request: Request, category: int):
        try:
            json: dict = request.json
            lst = json.get("list")
            danList: list = json.get("dans", [])
            select = (
                Select(sysDictPO.desc)
                .filter(sysDictPO.id.__eq__(category))
            )
            # dictOne: dict = await db_tools.execForMappings(self.get_db_session(request), select, True)
            db = QueryOneCallable(self.get_db_session(request))
            dictOne: dict = await db(select=select, isPO=False)
            desc: str = dictOne.get("desc", "")
            arr = re.split(r'\r\n|\r|\n', desc)
            arr = [a.replace(' ', '').split(',') for a in arr]
            rest = data.Padding(lst, arr, *danList)
            return self.response_json(Result.success({"list": rest, "count": len(rest)}))
        except Exception as e:
            return self.response_json(Result.fail(message=f"异常：{e}"))
