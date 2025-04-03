from __future__ import annotations
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
from co6co.utils import try_except
import asyncio
from co6co_sanic_ext import sanics
from services.tasks import custom

from abc import ABC, abstractmethod


class HandlerCommand(ABC):
    def __init__(self, successor: HandlerCommand = None, taskMgr: TasksMgr = None):
        self.successor = successor
        taskMgr = taskMgr or successor.taskMgr
        self.scheduler = taskMgr. scheduler if taskMgr else None
        self.taskMgr: TasksMgr = taskMgr

    def handle_request(self,  data: DATA, conn: PipeConnection):
        self.command: CommandCategory = data.command
        self.taskCode = data.code
        self.data = data
        self.conn = conn
        handled = self._handle()
        if not handled and self.successor:
            self.successor.handle_request(data, conn)

    def getSourceCode(self):
        sourceCode = self. data.sourceCode
        if not sourceCode:
            code = self. data.sourceForm
            task = custom.get_task(code)
            if task:
                sourceCode = task.main
        if not sourceCode:
            message = f"任务{self.taskCode}，未找到任务"
            self.sendData(False, message)
        return sourceCode

    def sendData(self, success: bool, message: str):
        resultData = CommandCategory.createOption(CommandCategory.GET, success=success, data=message)
        self.conn.send(resultData)

    def _taskisExist(self):
        result = self.scheduler.exist(self. taskCode)
        return result

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError('Must provide implementation in subclass.')

    @abstractmethod
    def _handle(self,   data: DATA, conn: PipeConnection) -> bool:
        """
        return 返回 True 已经被处理了
        """
        raise NotImplementedError('Must provide implementation in subclass.')


class ExistHandler(HandlerCommand):

    def check(self):
        return self.command == CommandCategory.Exist

    def _handle(self):
        if not self.check():
            return False
        try:
            result = self._taskisExist()
            message = f"任务'{self.taskCode}'->存在" if result else f"任务'{self.taskCode}'->不存在"
            self.sendData(result, message)
        except Exception as e:
            log.err("ExistHandler", e)
        return True


class StartHandler(HandlerCommand):
    def check(self):
        return self.command == CommandCategory.START

    def _handle(self):
        if not self.check():
            return False
        try:
            sourceCode = self.getSourceCode()
            if not sourceCode:
                return
            if not self._taskisExist():
                result, message = self.scheduler.addTask(self.taskCode, sourceCode, self. data.cron)
                fall_result = self.taskMgr.run(self.taskMgr.update_status, [self.taskCode], 1)
            else:
                result = False
                message = f"任务{self.taskCode}，已存在"
            self.sendData(result, message)
        except Exception as e:
            log.err("StartHandler", e)
        return True


class ModifyHandler(HandlerCommand):
    def check(self):
        return self.command == CommandCategory.MODIFY

    def _handle(self):
        if not self.check():
            return False
        try:
            sourceCode = self.getSourceCode()
            if not sourceCode:
                return
            if not self._taskisExist():
                result = False
                message = f"任务{self.taskCode}，不存在"
            else:
                result = self.scheduler.modifyTask(self.taskCode, sourceCode, self.data.cron)
                message = f"任务{self.taskCode}，修改成功" if result else f"任务{self.taskCode}，修改失败"
            self.sendData(result, message)
        except Exception as e:
            log.err("ModifyHandler", e)
        return True


class RemoveHandler(HandlerCommand):
    def check(self):
        return self.command == CommandCategory.REMOVE

    def _handle(self):
        if not self.check():
            return False
        try:
            if not self._taskisExist():
                result = False
                message = f"任务{self.taskCode}，不存在!"
            else:
                result = self.scheduler.removeTask(self. taskCode)
                if result:
                    fall_result = self.taskMgr.run(self.taskMgr.update_status, [self. taskCode], 0)
                    log.warn(f"任务{self.taskCode}，删除成功，更新状态：{fall_result}")
                    message = f"任务{self.taskCode}，删除成功"
                else:
                    message = f"任务{self.taskCode}，删除失败"
            self.sendData(result, message)
        except Exception as e:
            log.err("RemoveHandler", e)
        return True


class UnknownHandler(HandlerCommand):
    def check(self):
        return True

    def _handle(self):
        if not self.check():
            return False
        message = f"未处理命令{self.command.key}"
        self.sendData(False, message)
        log.warn(f"未处理命令{self.command.key},{self.taskCode}")
        return True


class TasksMgr(BaseBll, sanics.Worker):
    def __init__(self, app: Sanic, event: asyncio.Event, conn: PipeConnection):
        BaseBll.__init__(self, app=app)
        sanics.Worker.__init__(self, event, conn)
        app.ctx.taskMgr = self
        self.scheduler = Scheduler()
        self.handlerChain = ExistHandler(StartHandler(ModifyHandler(RemoveHandler(UnknownHandler(taskMgr=self)))))

    @try_except
    def handler(self, data: str | DATA, conn: PipeConnection):
        """
        处理数据
        """
        data: DATA = data
        self.handlerChain.handle_request(data, conn)

    async def getData(self):
        """
        获取源码
        """
        try:
            call = QueryListCallable(self.session)
            select = (
                Select(SysTaskPO.data, SysTaskPO.code, SysTaskPO.category, SysTaskPO.cron, DynamicCodePO.sourceCode)
                .outerjoin(DynamicCodePO, DynamicCodePO.id == SysTaskPO.data)
                .filter(SysTaskPO.state == dict_state.enabled.val)
            )
            return await call(select, isPO=False)

        except Exception as e:
            log.err("执行 ERROR", e)
            return []

    async def update_status(self, codeList: List[str] = None, status: int = 0) -> int:
        """
        更新状态
        codeList: 任务编码 None -->所有，[] --> 不更新，[,,,]--> 更新指定
        status: 状态 0: 停止 1:运行
        """
        try:
            if codeList and len(codeList) == 0:
                return 0
            if codeList == None:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val).values({SysTaskPO.execStatus: status})
            else:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val, SysTaskPO.code.in_(codeList)).values({SysTaskPO.execStatus: status})
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
        taskArr = self.run(self.getData)
        # data = asyncio.run(self.getData())
        # result = asyncio.run(self.check_session_closed())
        # log.warn(data)
        success = []
        faile = []
        for po in taskArr:
            code = po.get("code")
            category = po.get("category")
            cron = po.get("cron")
            sourceCode = po.get("sourceCode")
            item = po.get("data")  # 代码Id，或者 类code属性
            log.info("加载任务:{}...".format(code))
            if category == 0:
                # log.warn("任务在代码中，加找到加载下个模块：{}".format(code))
                task = custom. get_task(item)
                if task:
                    self.scheduler.addTask(code, task.main, cron)
                    success.append(code)
                else:
                    faile.append(code)
                continue
            # 任务在表中，已经关联表读取完成
            if self. scheduler.checkCode(sourceCode, cron):
                self. scheduler.addTask(code, sourceCode, cron)
                success.append(code)
                continue
            else:
                faile.append(code)
                log.warn("检查代码失败：{}".format(code))
        log.warn("加载任务完成,预加载：{}共加载,{}个任务".format(len(taskArr), self.scheduler.task_total))
        succ_result = 0
        fall_result = 0
        if len(success) > 0:
            print(*success)
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
