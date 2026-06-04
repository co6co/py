
from functools import wraps
from co6co.utils import log 
from ...services.configCache import ConfigCache
from ..base_view import AuthMethodView


def ConfigEntry(f):
    """
    缓存配置相关
    """
    @wraps(f)
    async def _function(self,*args, **kwargs):
        if isinstance(self, AuthMethodView):
            cacheManage = ConfigCache(self.request) 
        code = self.match_info.get("code") 
        value = await f(*args, **kwargs)
        if code is None:
            log.warn("code参数是必须的,当前请求参数:",self.match_info)
        elif "SYS_CONFIG" in code:
            if cacheManage is not None:
                value = await f(*args, **kwargs)
                cacheManage.setConfig(code, value)
            else:
                log.warn("cacheManage 未找到 Request 参数")
        return value

    return _function
