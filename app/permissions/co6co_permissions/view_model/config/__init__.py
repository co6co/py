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
from ..biz_view import AbsExistView, AbsSelectView

from ...model.filters.config_filter import Filter
from ...model.pos.other import sysConfigPO
from ..aop.config_aop import ConfigEntry
from ...services.configCache import ConfigCache


class ConfigView(AuthMethodView):
    """
    通过代码获取配置
    """

    routePath = "/<code:str>"

    def get_sql(self) -> Select:
        code=self.mapping.get("code")
        select = Select(
            sysConfigPO.name, sysConfigPO.code, sysConfigPO.value, sysConfigPO.remark
        ).filter(sysConfigPO.code.__eq__(code))
        return select

    async def get(self ):
        """
        获取配置
        code: 配置代码
        """
        select =self.get_sql()
        return await self.query_mapping( select, oneRecord=True)


class ConfigByCacheView(AuthMethodView):
    """
    通过代码获取配置
    """

    routePath = "/Cache/<code:str>"
    

    async def post(self ):
        """
        获取配置
        code: 配置代码

        return str,配置值
        """
        code=self.mapping.get("code")
        cache = ConfigCache(self.request)
        config = cache.getConfig(code)
        if not config:
            config = await cache.queryConfig(code)
        if config is None:
            return self.response_json(
                Result.fail(
                    message="未能获取配置，请检查是否有配置项'{}',或检查配置的json对象是否正确".format(
                        code
                    )
                )
            )
        return self.response_json(Result.success(config))


class ExistView(AbsExistView):
    @property
    def column(self):
        return sysConfigPO.id

    @property
    def exist_condition(self):
        return sysConfigPO.code == self.param_code, sysConfigPO.id != self.param_pk


@ConfigEntry
async def configValueChange(self: AuthMethodView, code: str, value: str):
    return value 
class Views(AuthMethodView):
    async def post(self ):
        """
        table数据
        """
        param = Filter()
        return await self.query_page( param)

    async def put(self ):
        """
        增加
        """
        po = sysConfigPO()
        userId = self.userId

        async def before(po: sysConfigPO, session: AsyncSession, request):
            exist = await db_tools.exist(
                session, sysConfigPO.code.__eq__(po.code), column=sysConfigPO.id
            )
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))

        async def after(po: sysConfigPO, session, request):
            await configValueChange(self, po.code, po.value)

        return await self.add(
             po, userId=userId, beforeFun=before, afterFun=after
        )


class View(AuthMethodView):
    routePath = "/<pk:int>"
    def pk(self)->int:
        return self.match_info.get("pk")

    async def put(self ):
        """
        编辑
        """

        async def before(
            oldPo: sysConfigPO, po: sysConfigPO, session: AsyncSession, request
        ):
            exist = await db_tools.exist(
                session,
                sysConfigPO.id != oldPo.id,
                sysConfigPO.code.__eq__(po.code),
                column=sysConfigPO.id,
            )
            if exist:
                return response_json(Result.fail(message=f"'{po.code}'已存在！"))
            await configValueChange(request, po.code, po.value)

        return await self.edit(self.pk, sysConfigPO, userId=self.userId, fun=before )

    async def delete(self ):
        """
        删除
        """
        return await self.remove(self.pk, sysConfigPO)
