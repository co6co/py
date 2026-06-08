
from sanic.response import text
from sanic import Request 
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co.data.result import Result
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from co6co_permissions.view_model.biz_view import AbsExistView 
from ...model.pos.tables import DynamicCodePO
from .._filters.dynamic import Filter

from .codeView import _codeView

class ExistView(AbsExistView): 
    @property
    def column(self):
        return DynamicCodePO.id

    @property 
    def exist_condition(self): 
        return DynamicCodePO.code == self.param_code, DynamicCodePO.id != self.param_pk 

class Views(AuthMethodView):
    async def get(self ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(DynamicCodePO.id, DynamicCodePO.name, DynamicCodePO.code, DynamicCodePO.state)
            .filter(DynamicCodePO.state == 1, DynamicCodePO.category == 1)
            .order_by(DynamicCodePO.code.asc())
        )
        return await self.query_list( select,  isPO=False)

    async def post(self ):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        param.__dict__.update(self.json)

        return await self.query_page(param)

    async def put(self ):
        """
        增加
        """
        po = DynamicCodePO()
        userId = self.userId
        po.__dict__.update(self.json)

        async def before(po: DynamicCodePO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  DynamicCodePO.code.__eq__(po.code), column=DynamicCodePO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")

        return await self.add( po, json2Po=False, userId=userId, beforeFun=before)


class View(AuthMethodView):
    routePath = "/<pk:int>"

    @property
    def pk(self):
        return self.match_info.get("pk")
    
    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: DynamicCodePO, po: DynamicCodePO, session: AsyncSession, request):
            exist = await db_tools.exist(session, DynamicCodePO.id != oldPo.id, DynamicCodePO.code.__eq__(po.code), column=DynamicCodePO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")
        return await self.edit( self.pk, DynamicCodePO,  userId=self.userId, fun=before)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, DynamicCodePO)


class RunView(_codeView, AuthMethodView):
    routePath = "/run/<pk:int>"
    @property
    def pk(self):
        return self.match_info.get("pk")
    async def put(self, request: Request, pk: int):
        """
        执行一次
        """
        select = Select(DynamicCodePO).filter(DynamicCodePO.id.__eq__(self.pk))
        po: DynamicCodePO = await self.get_one( select)
        return self.response_json(self.exec_py_code(po.sourceCode))
