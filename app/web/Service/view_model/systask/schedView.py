

from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Update
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from co6co_web_db.view_model import get_one
from model.pos.tables import TaskPO
from services.tasks import Scheduler
from services.tasks import CuntomCronTrigger
from datetime import datetime
from co6co_permissions.model.enum import dict_state


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
        cron 表达式 合法性检测        
        ?cron=0 0 0 12 12 *
        """
        data = self.usable_args(request)
        cron = data.get("cron", None)
        return await self._post(cron)

    async def post(self, request: Request):
        """
        cron 表达式 合法性检测 
        json:{cron:'0 0 0 12 12 *'}
        """
        json: dict = request.json
        cron = json.get("cron", None)
        return await self._post(cron)


class _codeView:
    def exec_py_code(self, pyCode: str):
        """
        运行PyCode
        """
        try:
            res, _e = Scheduler.parseCode(pyCode)
            if res:
                res = _e()
                return Result.success(data=res)
            else:
                return Result.fail(message="解析出错：{}".format(e))
        except Exception as e:
            return Result.fail(message="执行出错：{}".format(e))


class codeView(_codeView, AuthMethodView):
    routePath = "/code/test"

    async def post(self, request: Request):
        """
        检查代码 python 代码

        params: {code:'python 代码'}
        return {data:False|True,...}
        """
        json: dict = request.json
        code = json.get("code", None)
        res, _e = Scheduler.parseCode(code)
        if res:
            return self.response_json(Result.success(data=True))
        return self.response_json(Result.success(data=False, message="解析代码出错:'{}'".format(_e)))

    async def put(self, request: Request):
        """
        执行代码
        params: {code:'python 代码'}
        return {data:False|True,...}
        """
        json: dict = request.json
        code = json.get("code", None)
        return self.response_json(self.exec_py_code(code))


class schedView(_codeView, AuthMethodView):
    routePath = "/sched/<pk:int>"

    def getScheduler(self, request: Request) -> Scheduler:

        return request.app.ctx.scheduler

    async def post(self, request: Request, pk: int):
        """
        调度 
        周期性执行的可以执行完成的代码
        """
        select = Select(TaskPO).filter(TaskPO.id.__eq__(pk))
        po: TaskPO = await get_one(request, select)
        scheduler = self.getScheduler(request)
        exist = scheduler.exist(po.code)
        if exist:
            res = scheduler.addTask(po.code, po.sourceCode, po.cron)
        else:
            res = scheduler.modifyTask(po.code, po.sourceCode, po.cron)

        if res:
            return self.response_json(Result.success())
        else:
            return self.response_json(Result.fail())

    async def delete(self, request: Request, pk: int):
        """
        停止调度
        """
        select = Select(TaskPO).filter(TaskPO.id.__eq__(pk))
        po: TaskPO = await get_one(request, select)
        scheduler = self.getScheduler(request)
        result = scheduler.removeTask(po.code)

        if result:
            po.execStatus = 0
            session = self.get_db_session(request)
            await session.commit()
            return self.response_json(Result.success(message="停止成功"))
        else:
            return self.response_json(Result.fail(message="停止失败"))

    async def put(self, request: Request, pk: int):
        """
        执行一次
        """
        select = Select(TaskPO).filter(TaskPO.id.__eq__(pk))
        po: TaskPO = await get_one(request, select)
        return self.response_json(self.exec_py_code(po.sourceCode))
