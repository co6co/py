from model.enum import CommandCategory
from co6co.task.thread import ThreadEvent
from co6co_permissions.services.bll import BaseBll
from model.pos.tables import DynamicCodePO, SysTaskPO
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
        data: DATA = data
        command: CommandCategory = data.command
        result = False
        message: str = None
        taskCode = data.code
        if command == CommandCategory.Exist:
            result = self.scheduler.exist(taskCode)
            message = f"任务'{taskCode}'->存在" if result else f"任务'{taskCode}'->不存在"
        elif command == CommandCategory.START:
            result, message = self.scheduler.addTask(taskCode, data.sourceCode, data.cron)
            fall_result = self.run(self.update_status, [taskCode], 1)
        else:
            # 下面都是任务存在才能处理的命令
            if not self.scheduler.exist(taskCode):
                message = f"任务{taskCode}，不存在"
            else:
                log.warn("任务存在，开始处理命令：", command)
                if command == CommandCategory.REMOVE:
                    result = self.scheduler.removeTask(taskCode)
                    if result:
                        fall_result = self.run(self.update_status, [taskCode], 0)
                        log.warn(f"任务{taskCode}，删除成功，更新状态：{fall_result}")
                elif command == CommandCategory.MODIFY:
                    result = self.scheduler.modifyTask(taskCode, data.sourceCode, data.cron)
                else:
                    result = False
                    message = f"未处理命令{command.key}"
        resultData = CommandCategory.createOption(CommandCategory.GET, success=result, data=message)
        conn.send(resultData)
        log.succ(f"处理命令’{command.key}‘{taskCode}结果:{result}->{message}") if result else log.warn(f"处理命令{command.key},{taskCode}结果:{result}->{message}")

    async def getData(self):
        """
        获取源码
        """
        try:
            call = QueryListCallable(self.session)
            select = (
                Select(SysTaskPO.data, SysTaskPO.code, SysTaskPO.category, SysTaskPO.cron, DynamicCodePO.sourceCode)
                .outerjoin(DynamicCodePO, DynamicCodePO.id == SysTaskPO.data)
                .filter(DynamicCodePO.category == 1, DynamicCodePO.state == dict_state.enabled.val)
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
                ccc = Update(SysTaskPO).where(SysTaskPO.category == 1, SysTaskPO.state == dict_state.enabled.val).values({SysTaskPO.execStatus: status})
            else:
                ccc = Update(SysTaskPO).where(SysTaskPO.category == 1, SysTaskPO.state == dict_state.enabled.val, SysTaskPO.code.in_(codeList)).values({SysTaskPO.execStatus: status})
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
            category = po.get("category")
            cron = po.get("cron")
            sourceCode = po.get("sourceCode")
            data = po.get("data")
            log.info("加载任务:{}...".format(code))
            if category == 0:
                log.warn("任务在代码中，加找到加载下个模块：{}".format(code))
                continue
            # 任务在表中，已经关联表读取完成
            if self. scheduler.checkCode(sourceCode, cron):
                self. scheduler.addTask(code, sourceCode, cron)
                success.append(code)
                pass
            else:
                faile.append(code)
                log.warn("检查代码失败：{}".format(code))
        log.warn("加载任务完成,预加载：{}共加载{}个任务".format(len(data), self.scheduler.task_total))
        succ_result = 0
        fall_result = 0
        if len(success) > 0:
            succ_result = self.run(self.update_status, success, 1)
        if len(faile) > 0:
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
