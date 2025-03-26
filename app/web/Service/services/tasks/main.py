from co6co.task.thread import ThreadEvent
from co6co_permissions.services.bll import BaseBll
from model.pos.tables import TaskPO
from sanic import Sanic
from services.tasks import Scheduler
from sqlalchemy.sql import Select, Update
from co6co_db_ext.db_utils import db_tools, QueryListCallable
from typing import List
from co6co.utils import log
from co6co_permissions.model.enum import dict_state


class TasksMgr(BaseBll):

    def __init__(self, app: Sanic):
        super().__init__()
        self.app = app
        self.scheduler = Scheduler()

    def extend(self, app: Sanic, extendName=None, extendObj=None):
        """
        扩展sanic
        """
        if not hasattr(app.ctx, "extensions"):
            app.ctx.extensions = {}
        if extendName and extendObj:
            app.ctx.extensions[extendName] = extendObj

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

    def startTimeTask(self):
        """
        运行在数据库中的代码任务
        """
        data = self.run(self.getData)
        # data = asyncio.run(self.getData())
        # result = asyncio.run(self.check_session_closed())
        log.warn(data)
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

    def stop(self):
        result = self.run(self.update_status)
        log.warn("状态更新,成功->{}".format(result))
