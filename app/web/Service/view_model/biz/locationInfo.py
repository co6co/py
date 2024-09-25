
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.business import locationInforPO
from view_model._filters.locationInfo import Filter


class Views(AuthMethodView):
    async def get(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(locationInforPO.id, locationInforPO.name)
            .order_by(locationInforPO.createTime.asc())
        )
        return await self.query_list(request, select,  isPO=False)

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
        po = locationInforPO()
        userId = self.getUserId(request)

        async def beforeFun(po: locationInforPO, session, request):
            if not self.is_integer(po.category):
                return self.response_json(Result.fail(message="类型必须为为数值！"))

        return await self.add(request, po, userId=userId, beforeFun=beforeFun)

    def patch(self, request: Request):
        return text("I am patch method")


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        po = locationInforPO()
        po.__dict__.update(request.json)

        async def beforeFun(old, po: locationInforPO, session, request):
            if not self.is_integer(po.category):
                return self.response_json(Result.fail(message="类型必须为为数值！"))
        return await self.edit(request, pk, locationInforPO, po=po, userId=self.getUserId(request), fun=beforeFun)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, locationInforPO)
