from model.enum import CommandCategory
from co6co.task.thread import ThreadEvent
from co6co_permissions.services.bll import BaseBll
from model.pos.tables import TaskPO
from sanic import Sanic
from services.tasks import Scheduler
from sqlalchemy.sql import Select, Update
from co6co_db_ext.db_utils import db_tools, QueryListCallable
from typing import List
from co6co.utils import log, DATA
from co6co_permissions.model.enum import dict_state
from multiprocessing.connection import PipeConnection

import asyncio
from co6co_sanic_ext import sanics


class TasksMgr(BaseBll, sanics.Worker):
    def __init__(self, app: Sanic, event: asyncio.Event, conn: PipeConnection):
        BaseBll.__init__(self, app=app)
        sanics.Worker.__init__(self, event, conn)
        app.ctx.taskMgr = self
        self.scheduler = Scheduler()

    def handler(self, data: str, conn: PipeConnection):
        """
        处理数据
        """
        log.warn("接收到命令：", data)
        data: DATA = data
        command: CommandCategory = data.command
        result = False
        message: str = None
        if command == CommandCategory.Exist:
            result = self.scheduler.exist(data.data)
            message = f"任务{data.data}，存在" if result else f"任务{data.data}，不存在"
        # 下面都是任务存在才能处理的命令
        if not self.scheduler.exist(data.data):
            message = f"任务{data.data}，不存在"
        else:
            if command == CommandCategory.REMOVE:
                result = self.scheduler.removeTask(data.data)
            if command == CommandCategory.START:
                result = self.scheduler.addTask(data.code, data.sourceCode, data.cron)
            if command == CommandCategory.MODIFY:
                result = self.scheduler.modifyTask(data.code, data.sourceCode, data.cron)
            else:
                result = False
                message = f"未处理命令{command.key}"
        resultData = CommandCategory.createOption(CommandCategory.GET, success=result, data=message)
        conn.send(resultData)
        log.succ(f"处理命令{data.data}结果", result) if result else log.warn(f"处理命令{data.data}结果", result)

    async def getData(self):
        """
        获取源码
        """
        try:
            call = QueryListCallable(self.session)
            select = (
                Select(TaskPO.sourceCode, TaskPO.code, TaskPO.cron)
                .filter(TaskPO.category == 1, TaskPO.state == dict_state.enabled.val)
            )
            return await call(select, isPO=False)

        except Exception as e:
            log.err("执行 ERROR", e)
            return []

    async def update_status(self, codeList: List[str] = None, status: int = 0) -> int:
        """
        更新状态
        codeList: 任务编码 None -->所有，[] --> 不更新，[,,,]--> 更新指定
        status: 状态 0: 停止，1:运行
        """
        try:
            if codeList and len(codeList) == 0:
                return 0
            if codeList == None:
                ccc = Update(TaskPO).where(TaskPO.category == 1, TaskPO.state == dict_state.enabled.val).values({TaskPO.execStatus: status})
            else:
                ccc = Update(TaskPO).where(TaskPO.category == 1, TaskPO.state == dict_state.enabled.val, TaskPO.code.in_(codeList)).values({TaskPO.execStatus: status})
            result = await db_tools.execSQL(self.session, ccc)
            await self.session.commit()
            return result

        except Exception as e:
            log.err("执行 ERROR", e)
            return None

    def _startTimeTask(self):
        """
        运行在数据库中的代码任务
        """
        data = self.run(self.getData)
        # data = asyncio.run(self.getData())
        # result = asyncio.run(self.check_session_closed())
        # log.warn(data)
        success = []
        faile = []
        for po in data:
            code = po.get("code")
            sourceCode = po.get("sourceCode")
            cron = po.get("cron")
            log.info("加载任务:{}...".format(code))
            if self. scheduler.checkCode(sourceCode, cron):
                self. scheduler.addTask(code, sourceCode, cron)
                success.append(code)
                pass
            else:
                faile.append(code)
                log.warn("检查代码失败：{}".format(code))
        log.warn("加载任务完成,预加载：{}共加载{}个任务".format(len(data), self.scheduler.task_total))
        succ_result = self.run(self.update_status, success, 1)
        fall_result = self.run(self.update_status, faile, 0)
        log.warn("状态更新,成功->{},失败->{}".format(succ_result, fall_result))

    def start(self):
        """
        启动任务
        """
        super().start()
        self._startTimeTask()
        pass

    def stop(self):
        super().stop()
        result = self.run(self.update_status)
        log.warn("状态更新,成功->{}".format(result))
