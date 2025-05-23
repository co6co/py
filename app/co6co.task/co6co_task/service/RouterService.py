# -*- coding:utf-8 -*-

from co6co_db_ext.session import BaseBll
from sqlalchemy import Select
from co6co.utils import log
from co6co_db_ext.db_utils import QueryListCallable
from co6co_permissions.model.enum import dict_state
from sanic import Sanic
from co6co_sanic_ext import sanics
from ..model.pos.tables import DynamicCodePO


class dynamicRouter(BaseBll):
    """
    动态路由
    """
    @staticmethod
    def appendRoute(app: Sanic):
        try:
            bll = dynamicRouter(app.config.db_settings)
            source = bll.getSourceList()
            sanics.App.appendView(app, *source, blueName="user_append_View")
            pass
        except Exception as e:
            log.err("动态模块失败", e)

    async def _getSourceList(self):
        """
        获取源码
        """
        try:
            call = QueryListCallable(self.session)
            select = (
                Select(DynamicCodePO.sourceCode)
                .filter(DynamicCodePO.category == 2, DynamicCodePO.state == dict_state.enabled.val)
            )
            result = await call(select, isPO=False)
            return [item.get("sourceCode") for item in result]

        except Exception as e:
            log.err("执行 ERROR", e)
            return []
        finally:
            # await self.session.reset()
            pass

    def getSourceList(self):
        return self.run(self._getSourceList)
