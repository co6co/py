from co6co.data.result import Result 
from co6co.utils import log
from ..base_view import AbsClsView
from ...services .configCache import ConfigCache

class UI_Config_View(AbsClsView):
    """
    不需要认证
    获取UI基础配置
    """
    routePath = "/ui" 
    async def post(self ):
        """ 
        获取配置
        code: 配置代码

        return str,配置值
        """
        cache = ConfigCache(self.request)
        code="SYS_CONFIG_BASE_UI"
        config = cache.getConfig(code)
        if not config:
            config = await cache .queryConfig(code)
        if config is None: 
            log.warn(f"未能获取配置项'{code}'，请检查是否有配置项'{code}',或检查配置的json对象是否正确")
            return self.response_json(Result.fail(message="未能获取配置，请检查是否有配置项'{}',或检查配置的json对象是否正确".format(code)))
        return self.response_json(Result.success(config))

