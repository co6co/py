from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView 
from co6co_permissions.view_model.biz_view import AbsExistView ,AbsPkView
from sqlalchemy.sql import Select 

from ...model.pos.tables import SysTaskPO, DynamicCodePO
from .._filters.sysTask import Filter
from ...service import CustomTask as custom


class ExistView(AbsExistView): 
    @property
    def column(self):
        return SysTaskPO.id

    @property 
    def exist_condition(self): 
        return SysTaskPO.code == self.param_code, SysTaskPO.id != self.param_pk 

class SelectViews(AuthMethodView):
    routePath = "/select/<category:int>"

    async def get(self):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        category=self.match_info.get("category", 0)
        if category == 0:
            data = custom.get_list()
            all = [{"id": t[1], "name": t[0]}for t in data]
            return self.response_json(Result.success(data=all))
        else:
            select = (
                Select(DynamicCodePO.id, DynamicCodePO.name)
                .filter(DynamicCodePO.state == 1, DynamicCodePO.category == 1)
                .order_by(DynamicCodePO.code.asc())
            )
            return await self.query_list( select,  isPO=False)


class Views(AuthMethodView):
    async def get(self ):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(SysTaskPO.id, SysTaskPO.name, SysTaskPO.code, SysTaskPO.state, SysTaskPO.execStatus)
            .order_by(SysTaskPO.code.asc())
        )
        return await self.query_list( select,  isPO=False)

    async def post(self ):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        param.__dict__.update(self.json)

        return await self.query_page( param)

    async def put(self ):
        """
        增加
        """
        po = SysTaskPO() 
        po.__dict__.update(self.json) 
        async def before(po: SysTaskPO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  SysTaskPO.code.__eq__(po.code), column=SysTaskPO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")

        return await self.add( po, json2Po=False, userId= self.userId, beforeFun=before)


class View(AbsPkView): 
    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: SysTaskPO, po: SysTaskPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, SysTaskPO.id != oldPo.id, SysTaskPO.code.__eq__(po.code), column=SysTaskPO.id)
            if exist:
                return Result.fail(message=f"'{po.code}'已存在！")
            if oldPo.execStatus == 1:
                return Result.fail(message="任务正在运行中请先停止，后再进行编辑！")

        return await self.edit(self.routeValue, SysTaskPO,  userId=self.userId, fun=before)

    async def delete(self ):
        """
        删除
        """
        return await self.remove(self.routeValue, SysTaskPO)
