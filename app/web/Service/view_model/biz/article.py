
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.business import articlePO
from view_model._filters.article import Filter


class Views(AuthMethodView):

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = articlePO()
        userId = self.getUserId(request)
        po.__dict__.update(request.json)

        async def before(po: articlePO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  articlePO.code.__eq__(po.code), column=articlePO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))

        return await self.add(request, po, json2Po=False, userId=userId, beforeFun=before)


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        po = articlePO()
        po.__dict__.update(request.json)

        async def before(oldPo: articlePO, po: articlePO, session: AsyncSession, request):
            exist = await db_tools.exist(session, articlePO.id != oldPo.id, articlePO.code.__eq__(po.code), column=articlePO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.edit(request, pk, articlePO, po=po, userId=self.getUserId(request), fun=before)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, articlePO)
