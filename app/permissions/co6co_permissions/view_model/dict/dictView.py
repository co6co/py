
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Delete

from co6co_db_ext.db_utils import db_tools, DbCallable
from co6co_web_db.model.params import associationParam

from datetime import datetime
from ..base_view import AuthMethodView
from ...model.filters.dict_filter import DictFilter
from ...model.pos.other import sysDictPO


class Views(AuthMethodView):
    async def post(self, request: Request):
        """
        table数据 
        """
        param = DictFilter()
        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = sysDictPO()
        userId = self.getUserId(request)

        async def before(po: sysDictPO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  sysDictPO.code.__eq__(po.code), column=sysDictPO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.add(request, po, userId=userId, beforeFun=before)


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        async def before(oldPo: sysDictPO, po: sysDictPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, sysDictPO.id != oldPo.id, sysDictPO.code.__eq__(po.code), column=sysDictPO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))

        return await self.edit(request, pk, sysDictPO, userId=self.getUserId(request), fun=before)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, sysDictPO)
