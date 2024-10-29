
from co6co_db_ext.session import BaseBll
from model.pos.tables import TaskPO
from sqlalchemy import Select
from co6co.utils import log
from co6co_db_ext.db_utils import QueryListCallable


class TaskBll(BaseBll):
    async def _getSourceList(self):
        """
        获取源码
        """
        try:
            call = QueryListCallable(self.session)
            select = (
                Select(TaskPO.sourceCode)
                .filter(TaskPO.category == 2, TaskPO.state == 0)
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
