
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Delete

from co6co_db_ext.db_utils import db_tools, DbCallable
from co6co_web_db.model.params import associationParam

from datetime import datetime
from ..base_view import AuthMethodView 
from ..biz_view import AbsPkView,AbsExistView 
from ...model.filters.dict_type_filter import Filter
from ...model.filters.dict_filter import DictFilter

from ...model.enum import dict_state
from ...model.pos.other import sysDictTypePO, sysDictPO


class DictTypeExistView(AbsExistView): 
    @property
    def column(self):
        return sysDictTypePO.id

    @property
    def exist_condition(self) :
        return sysDictTypePO.code == self.param_code, sysDictTypePO.id != self.param_pk 


class DictTypeViews(AuthMethodView):
    routePath = "/type"

    async def get(self ):
        """
        字典类型 下拉 
        """
        select = (
            Select(sysDictTypePO.id, sysDictTypePO.name, sysDictTypePO.code)
            .filter(sysDictTypePO.state.__eq__(dict_state.enabled.val))
            .order_by(sysDictTypePO.order.asc())
        )
        return await self.query_list( select,  isPO=False)

    async def post(self):
        """
        table数据
        """
        param = Filter()
        return await self.query_page( param)

    async def put(self ):
        """
        增加
        """
        po = sysDictTypePO()
        userId = self.userId

        async def before(po: sysDictTypePO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  sysDictTypePO.code.__eq__(po.code), column=sysDictTypePO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.add( po, userId=userId, beforeFun=before)


class DictTypeView(AbsPkView):
    routePath = "/type/<pk:int>"

    async def get(self ):
        """
        获取字典选择
        """

        select = (
            Select(sysDictPO.id, sysDictPO.name,
                   sysDictPO.value, sysDictPO.desc)
            .filter(sysDictPO.dictTypeId.__eq__(self.routeValue), sysDictPO.state.__eq__(dict_state.enabled.val))
            .order_by(sysDictPO.order.asc())
        )
        return await self.query_list( select,  isPO=False)

    async def post(self):
        """
        获取字典,table数据
        """
        param = DictFilter(self.routeValue)
        return await self.query_page( param)

    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: sysDictTypePO, po: sysDictTypePO, session: AsyncSession, request):
            exist = await db_tools.exist(session, sysDictTypePO.id != oldPo.id, sysDictTypePO.code.__eq__(po.code), column=sysDictTypePO.id)
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))

        return await self.edit( self.routeValue, sysDictTypePO, userId=self.userId, fun=before)

    async def delete(self ):
        """
        删除
        """
        async def before(po: sysDictTypePO, session: AsyncSession):
            count = await db_tools.count(session, sysDictPO.dictTypeId == po.id, column=sysDictTypePO.id)
            if count > 0:
                return response_json(Result.fail(message=f"该'{po.name}'关联了字典不能删除！"))

        return await self.remove( self.routeValue, sysDictTypePO, beforeFun=before)
