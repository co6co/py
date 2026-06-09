
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select

from co6co_db_ext.db_utils import db_tools  

from co6co_db_ext .actuator import Actuator 
from ..base_view import AuthMethodView
from ..biz_view import AbsExistView,AbsQueryAndAddView, AbsPkView

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
        code=self.match_info.get("code")
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
        code=self.match_info.get("code")
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
async def configValueChange(self: AuthMethodView, code: str, value: str|None):
    return value 


class Views(AbsQueryAndAddView):
    routePath='/'
    cls=Filter 
    poCls=sysConfigPO 
    async def add_before(self,po,actuator: Actuator):
        exist=await actuator.exist( sysConfigPO.code.__eq__(po.code), column=sysConfigPO.id)
        if exist:
            return Result.fail(message=f"'{po.code}'已存在！")
    async def add_after(self,po,actuator: Actuator):
        await configValueChange(self, po.code, po.value)

    @property       
    def add_option(self):
        option=super().add_option
        option.beforeFun=self.add_before
        option.afterFun=self.add_after
        return option
        
class View(AbsPkView):
    cls=sysConfigPO

    async def edit_check_data(self,po: sysConfigPO, actuator: Actuator):
        exist= await actuator.exist( sysConfigPO.id != po.id,
                sysConfigPO.code.__eq__(po.code),
                column=sysConfigPO.id,)
        if exist:
                return Result.fail(message=f"'{po.code}'已存在！")
        await configValueChange(self, po.code, po.value)
        pass
    @property
    def edit_option(self): 
        option=super().edit_option
        option.beforeFun=self.edit_check_data 
        return option 
    async def delete_after_data(self,po: sysConfigPO, actuator: Actuator): 
        await configValueChange(self, po.code, None)
        pass
    @property
    def delete_option(self): 
        option=super().delete_option
        option.afterFun=self.delete_after_data
        return option 
     
