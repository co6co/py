
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.wx import WxSubMenuPO
from view_model._filters.wxSubMenu import Filter
from co6co_permissions.model.enum import dict_state


class Views(AuthMethodView):
    async def get(self, request: Request):
        """
        选择下拉框数据 
        """
        args = self.usable_args(request)
        category = args.get('category')

        if category == None or not self.is_integer(category):
            return self.response_json(Result.fail("category参数不正确或为空"))

        select = (
            Select(WxSubMenuPO.id, WxSubMenuPO.name, WxSubMenuPO.icon,
                   WxSubMenuPO.color, WxSubMenuPO.url)
            .filter(WxSubMenuPO.state.__eq__(dict_state.enabled.val), WxSubMenuPO.category.__eq__(category))
            .order_by(WxSubMenuPO.order.asc())
        )
        return await self.query_list(request, select,  isPO=False)

    async def post(self, request: Request):
        """
        树形 table数据
        table
        """
        param = Filter()
        param.__dict__.update(request.json)

        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = WxSubMenuPO()
        userId = self.getUserId(request)
        return await self.add(request, po,   userId=userId)


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        return await self.edit(request, pk, WxSubMenuPO,  userId=self.getUserId(request))

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, WxSubMenuPO)
