
from sanic import Request 
from co6co.data.result import Result 

from sqlalchemy.sql import Select 
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView 
from ...model.pos.tables import DynamicCodePO, SysTaskPO 
from co6co.utils import log, DATA

from multiprocessing.connection import PipeConnection
from ...model.enum import CommandCategory
from .codeView import _codeView
from ...service import CustomTask as custom
from co6co.task.pools import timeout


class schedView(_codeView, AuthMethodView):
    routePath = "/sched/<pk:int>"
    @property
    def pk(self):
        return self.match_info.get("pk")

    async def read_data(self ):
        """
        从数据库中读取 code ,sourceCode,cron
        """
        select = (
            Select(SysTaskPO.code, SysTaskPO.category, SysTaskPO.cron, DynamicCodePO.sourceCode, SysTaskPO.data)
            .outerjoin(DynamicCodePO, DynamicCodePO.id == SysTaskPO.data)
            .filter(SysTaskPO.id.__eq__(self.pk))
        )

        poDict: dict = await db_tools.execForMappings(self.db_session, select, queryOne=True)
        code = poDict.get("code")
        sourceCode = poDict.get("sourceCode")
        data = poDict.get("data")
        log.warn("sourceCode:", sourceCode)
        cron = poDict.get("cron")
        return code, cron, sourceCode, data

    @staticmethod
    def getPipConn(request: Request) -> PipeConnection:
        return request.app.ctx.child_conn

    async def post(self ):
        """
        调度
        周期性执行的可以执行完成的代码
        """
        code,  cron, sourceCode, data = await self.read_data( )
        conn = schedView.getPipConn(self.request)
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

    @staticmethod
    async def execCommand(request: Request, code: str, command: CommandCategory):
        """
        执行一些简单命令
        """
        conn = schedView.getPipConn(request)
        conn.send(CommandCategory.createOption(command, code=code))
        result: DATA = conn.recv()
        if result.success:
            return True, result.data
        else:
            return False, result.data

    async def patch(self ):
        """
        查询下一次运行时间
        """
        select = Select(SysTaskPO).filter(SysTaskPO.id.__eq__(self.pk))
        po: SysTaskPO = await self.actuator.query_one_entity( select)
        success, data = await schedView.execCommand(self.request, po.code, CommandCategory.GETNextTime)
        if success:
            return self.response_json(Result.success(data, message="获取成功"))
        else:
            return self.response_json(Result.fail(message="获取失败:"+data))

    async def delete(self ):
        """
        停止调度
        """
        select = Select(SysTaskPO).filter(SysTaskPO.id.__eq__(self.pk))
        po: SysTaskPO = await self.actuator.query_one_entity( select)
        success, msg = await schedView.execCommand(self.request, po.code, CommandCategory.STOP)
        if success:
            return self.response_json(Result.success(message="停止成功"))
        else:
            return self.response_json(Result.fail(message="停止失败:"+msg))

    async def put(self ):
        """
        执行一次
        """
        _,  _, sourceCode, data = await self.read_data( )
        isTimeout = False
        secounds = 4
        result = Result.success()
        if not sourceCode:
            try:
                task = custom.ICustomTask.createInstance(data) 
                isTimeout, _ = timeout(secounds, task.main)
                log.warn(f"执行任务是否超时：{isTimeout}")
            except Exception as e:
                log.err("执行失败", e)
                return self.response_json(Result.fail(e, message="执行失败"))
        else:
            isTimeout, result = timeout(secounds,  self.exec_py_code, False, sourceCode)
        if isTimeout:
            return self.response_json(Result.success(data="执行时间过长", message="执行时间过长！"))
        else:
            return self.response_json(result)
