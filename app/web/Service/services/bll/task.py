
from services.bll import baseBll, BaseBll
from model.pos.tables import TaskPO
from sqlalchemy import Select
from co6co.utils import log
from co6co_db_ext.db_utils import QueryListCallable


class TaskBll(BaseBll):
    async def getSourceList(self):
        """
        获取需订阅告警用户
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
            return []
