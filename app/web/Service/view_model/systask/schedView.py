

from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from co6co_web_db.view_model import get_one
from model.pos.tables import TaskPO
from services.tasks import Scheduler
from services.tasks import CuntomCronTrigger
from datetime import datetime


class cronViews(AuthMethodView):
    routePath = "/cron/test"

    async def _post(self, cron: str):
        if cron == None:
            return Result.success(data=False, message="cron 必须填写")
        else:
            try:
                x = CuntomCronTrigger.resolvecron(cron)
                now = datetime.now()
                next = x.get_next_fire_time(now, now)
                return self.response_json(Result.success(data=True, message="解析成功：当前时间:{},下载执行时间：{}".format(now, next)))
            except Exception as e:
                return self.response_json(Result.success(data=False, message="解析'{}'出错:'{}'".format(cron, e)))

    async def get(self, request: Request):
        """
        ?cron=0 0 0 12 12 *
        """
        data = self.usable_args(request)
        cron = data.get("cron", None)
        return await self._post(cron)

    async def post(self, request: Request):
        """
        {
            cron:'0 0 0 12 12 *'
        }
        """
        json: dict = request.json
        cron = json.get("cron", None)
        return await self._post(cron)


class codeView(AuthMethodView):
    routePath = "/code/test"

    async def post(self, request: Request):
        """
        {
            code:'python 代码'
        }
        """
        json: dict = request.json
        code = json.get("code", None)
        res, _e = Scheduler.parseCode(code)
        if res:
            return self.response_json(Result.success(data=True))
        return self.response_json(Result.success(data=False, message="解析代码出错:'{}'".format(_e)))


class schedView(AuthMethodView):
    routePath = "/sched/<pk:int>"

    async def post(self, request: Request, pk: int):
        """
        调度
        """
        """
        执行一次
        """
        select = Select(TaskPO).filter(TaskPO.id.__eq__(pk))
        po: TaskPO = await get_one(request, select)
        scheduler: Scheduler = Scheduler()
        res = scheduler.addTask(po.sourceCode, po.cron)
        if res:
            return self.response_json(Result.success())
        else:
            return self.response_json(Result.fail())

    async def put(self, request: Request, pk: int):
        """
        执行一次
        """
        select = Select(TaskPO).filter(TaskPO.id.__eq__(pk))
        po: TaskPO = await get_one(request, select)
        res, main_e = Scheduler.parseCode(po.sourceCode)
        if res:
            res = main_e()
            return self.response_json(Result.success(data=res))
        else:
            return self.response_json(Result.fail(message=main_e))