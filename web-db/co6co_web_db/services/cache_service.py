from sanic import Request
from co6co_db_ext.cacheManage import CacheManage
from . import get_cache, get_db_service
from sanic import Sanic
from co6co_db_ext.appconfig import AppConfig


class SanicCache(CacheManage):
    """
    session: 未创建，请需要是时再创建 
    """
    def __init__(self) -> None:
        app = Sanic.get_app()
        cache = get_cache(app)
        self.config = AppConfig.get_config(app.config) 
        super().__init__(cache, db_service=get_db_service(app)) 
    @property
    def appConfig(self):
        """
        当前用户ID
        """
        return self.config
