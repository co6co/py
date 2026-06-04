from typing import Optional

from sanic import Sanic
from co6co_db_ext.db_session import connectSetting
from co6co_db_ext.session import dbBll
from co6co_db_ext.appconfig import AppConfig


class BaseBll(dbBll):
    def __init__(
        self, *, db_settings: Optional[connectSetting] = {}, app: Sanic = None
    ) -> None:
        if db_settings is None:
            app = app or Sanic.get_app()
            config = AppConfig.get_config(app.config)
            db_settings = config.db
        super().__init__(db_settings=db_settings)
