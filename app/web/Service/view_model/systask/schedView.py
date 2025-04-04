from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Update
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from co6co_web_db.view_model import get_one
from model.pos.tables import DynamicCodePO, SysTaskPO
from datetime import datetime
from co6co_permissions.model.enum import dict_state
from co6co.utils import log, DATA

from multiprocessing.connection import PipeConnection
from model.enum import CommandCategory
from view_model.systask.codeView import _codeView
from services.tasks import custom


class schedView(_codeView, AuthMethodView):
    routePath = "/sched/<pk:int>"

    async def read_data(self, pk: int, request: Request):
        """
        从数据库中读取 code ,sourceCode,cron
        """
        select = (
            Select(SysTaskPO.code, SysTaskPO.category, SysTaskPO.cron, DynamicCodePO.sourceCode, SysTaskPO.data)
            .outerjoin(DynamicCodePO, DynamicCodePO.id == SysTaskPO.data)
            .filter(SysTaskPO.id.__eq__(pk))
        )

        poDict: dict = await db_tools.execForMappings(self.get_db_session(request), select, queryOne=True)
        code = poDict.get("code")
        sourceCode = poDict.get("sourceCode")
        data = poDict.get("data")
        log.warn("sourceCode:", sourceCode)
        cron = poDict.get("cron")
        return code, cron, sourceCode, data

    def getPipConn(self, request: Request) -> PipeConnection:
        return request.app.ctx.child_conn

    async def post(self, request: Request, pk: int):
        """
        调度
        周期性执行的可以执行完成的代码
        """
        code,  cron, sourceCode, data = await self.read_data(pk, request)
        conn = self.getPipConn(request)
        conn.send(CommandCategory.createOption(CommandCategory.Exist, code=code))
        result: DATA = conn.recv()
        param = {"code": code, "sourceCode": sourceCode, "sourceForm": data, "cron": cron}
        if result.success:
            param.update({"command": CommandCategory.MODIFY})
        else:
            param.update({"command": CommandCategory.START})

        conn.send(CommandCategory.createOption(**param))
        result: DATA = conn.recv()
        res = result.success
        if res:
            return self.response_json(Result.success())
        else:
            return self.response_json(Result.fail(message=result.data))

    async def delete(self, request: Request, pk: int):
        """
        停止调度
        """
        select = Select(SysTaskPO).filter(SysTaskPO.id.__eq__(pk))
        po: SysTaskPO = await get_one(request, select)
        conn = self.getPipConn(request)
        conn.send(CommandCategory.createOption(CommandCategory.REMOVE, code=po.code))
        result: DATA = conn.recv()
        if result.success:
            return self.response_json(Result.success(message="停止成功"))
        else:
            return self.response_json(Result.fail(message="停止失败:"+result.data))

    async def put(self, request: Request, pk: int):
        """
        执行一次
        """
        _,  _, sourceCode, data = await self.read_data(pk, request)
        if not sourceCode:
            try:
                print(data)
                task = custom.get_task(data)
                print(type(task), data)
                task.main()
                return self.response_json(Result.success("None", message="执行成功"))
            except Exception as e:
                log.err("执行失败", e)
                return self.response_json(Result.fail(e, message="执行失败"))
        return self.response_json(self.exec_py_code(sourceCode))
