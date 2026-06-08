
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select 

from co6co_db_ext.db_utils import db_tools 
from ..base_view import AuthMethodView
from ..biz_view import AbsPkView
from ...model.filters.dict_filter import DictFilter
from ...model.pos.other import sysDictPO, sysDictTypePO
from ...model.enum import dict_state


class DictSelectView(AuthMethodView):
    routePath = "/<dictTypeCode:str>/<category:int>"
    def __init__(self, request: Request, dictTypeCode: str, category: int, *args, **kwargs) -> None:
        super().__init__(request, *args, **kwargs)
        self.dictTypeCode = dictTypeCode
        self.category = category

    async def get(self ):
        """ 
        获取字典选择
        dictTypeCode: 字典类型代码
        """
        # NameValueFlag = 0,
        # NameValue = 1,
        # NameFlag = 2,
        # All = 999,
        fields = [sysDictPO.id, sysDictPO.name, sysDictPO.flag,  sysDictPO.value]
        if self.category == 1:
            fields = [sysDictPO.id, sysDictPO.name,  sysDictPO.value]
        if self.category == 2:
            fields = [sysDictPO.id, sysDictPO.name,  sysDictPO.flag]
        if self.category == 999:
            fields = [sysDictPO.id, sysDictPO.name, sysDictPO.flag, sysDictPO.value, sysDictPO.desc]

        select = (
            Select(*fields)
            .join(sysDictTypePO, onclause=sysDictPO.dictTypeId == sysDictTypePO.id)
            .filter(sysDictTypePO.code.__eq__(self.dictTypeCode), sysDictPO.state.__eq__(dict_state.enabled.val))
            .order_by(sysDictPO.order.asc())
        )
        return await self.query_list(select,  isPO=False)


class Views(AuthMethodView):
    async def get(self ):
        """
        字典、字典类型状态
        枚举类型 : dict_state
        """
        return response_json(Result.success(data=dict_state.to_dict_list()))

    async def post(self ):
        """
        table数据 
        """
        param = DictFilter()
        return await self.query_page(  param)

    async def put(self  ):
        """
        增加
        """
        po = sysDictPO()
        userId = self.userId

        async def before(po: sysDictPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, sysDictPO.dictTypeId == po.dictTypeId, sysDictPO.value == po.value,   column=sysDictPO.id)
            if exist:
                return Result.fail(message=f"值'{po.value}'在该字典中已存在！")
        return await self.add( po, userId=userId, beforeFun=before)


class View(AbsPkView): 
    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: sysDictPO, po: sysDictPO, session: AsyncSession,*args,**kwargs):
            exist = await db_tools.exist(session, sysDictPO.dictTypeId == po.dictTypeId, sysDictPO.value == po.value, sysDictPO.id != oldPo.id, column=sysDictPO.id)
            if exist:
                return Result.fail(message=f"'{po.value}'在该字典中已存在！")

        return await self.edit(self.routeValue, sysDictPO, userId=self.userId, fun=before)

    async def delete(self):
        """
        删除
        """
        return await self.remove(self.routeValue, sysDictPO)
