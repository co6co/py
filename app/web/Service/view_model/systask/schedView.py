

from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import TaskPO
from services.tasks import Scheduler


class schedView(AuthMethodView):
    routePath = "/sched/<pk:int>"

    async def post(self, request: Request, pk: int):
        """
        调度
        """
        scheduler: Scheduler = Scheduler()

        async def before(oldPo: TaskPO, po: TaskPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, TaskPO.id != oldPo.id, TaskPO.code.__eq__(po.code), column=TaskPO.id)
            scheduler.addTask(oldPo.sourceCode, oldPo.cron)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.edit(request, pk, TaskPO,  userId=self.getUserId(request), fun=before)

    async def put(self, request: Request, pk: int):
        """
        执行一次
        """
        async def before(oldPo: TaskPO, po: TaskPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, TaskPO.id != oldPo.id, TaskPO.code.__eq__(po.code), column=TaskPO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.edit(request, pk, TaskPO,  userId=self.getUserId(request), fun=before)
