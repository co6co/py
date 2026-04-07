from co6co.task.eventDispatcher import EventHandler, Event, EventType
from typing import Optional
import time
from co6co_sanic_ext .sanics import IWorker
from sanic import Sanic
from co6co.utils import log
from .. import Scheduler
from ..CustomTask import ICustomTask
from co6co_web_db.services.db_service import BaseBll
from co6co_db_ext.db_utils import db_tools
from ..model.pos.tables import DynamicCodePO, SysTaskPO
from co6co_permissions.model.enum import dict_state
from sqlalchemy.sql import Select, Update
from typing import List, Tuple 

class BaseTaskHandler(EventHandler):
    """任务处理器基类"""
    def __init__(self, app: Sanic):
        self.app = app
        self.scheduler = Scheduler()
        self.bll = BaseBll(app=app)
        self.session = self.bll.session
    
    def getSourceCode(self, data: dict) -> Tuple[str | callable, callable | None]:
        sourceCode = data.get('sourceCode')
        stop = None
        if not sourceCode:
            code = data.get('sourceForm')
            task = ICustomTask.createInstance(code)
            if task:
                sourceCode = task.main
                stop = task.stop
        if not sourceCode:
            message = f"任务{data.get('code')}，未找到任务"
            return None, message
        return sourceCode, stop
    
    def taskisExist(self, taskCode: str) -> bool:
        return self.scheduler.exist(taskCode)
    
    def update_status(self, codeList: List[str] = None, status: int = 0) -> int:
        """
        更新状态
        codeList: 任务编码 None -->所有，[] --> 不更新，[,,,]--> 更新指定
        status: 状态 0: 停止 1:运行
        """
        try:
            if codeList and len(codeList) == 0:
                return 0
            if codeList is None:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val).values({SysTaskPO.execStatus: status})
            else:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val, SysTaskPO.code.in_(codeList)).values({SysTaskPO.execStatus: status})
            result = self.bll.run(db_tools.execSQL, self.session, ccc)
            self.bll.run(self.session.commit)
            return result
        except Exception as e:
            log.err("执行 ERROR", e)
            return None


class ExistHandler(BaseTaskHandler):
    """存在任务处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):
            try:
                taskCode = event.data.get('code')
                result = self.taskisExist(taskCode)
                message = f"任务'{taskCode}'->存在" if result else f"任务'{taskCode}'->不存在"
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': result, 'message': message},
                    source=self.name,
                    timestamp=time.time()
                )
            except Exception as e:
                log.err("ExistHandler", e)
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': False, 'message': str(e)},
                    source=self.name,
                    timestamp=time.time()
                )
    
    @property
    def supported_events(self):
        return ['task_exist']


class StartHandler(BaseTaskHandler):
    """启动任务处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):
            try:
                data = event.data
                taskCode = data.get('code')
                sourceCode, stop = self.getSourceCode(data)
                if not sourceCode:
                    return Event(
                        event_type=EventType.RESULT,
                        data={'success': False, 'message': f"任务{taskCode}，未找到任务"},
                        source=self.name,
                        timestamp=time.time()
                    )
                if not self.taskisExist(taskCode):
                    result, message = self.scheduler.addTask(taskCode, sourceCode, data.get('cron'), stop)
                    if result:
                        self.update_status([taskCode], 1)
                else:
                    result = False
                    message = f"任务{taskCode}，已存在"
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': result, 'message': message},
                    source=self.name,
                    timestamp=time.time()
                )
            except Exception as e:
                log.err("StartHandler", e)
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': False, 'message': str(e)},
                    source=self.name,
                    timestamp=time.time()
                )
    
    @property
    def supported_events(self):
        return ['task_start']


class ModifyHandler(BaseTaskHandler):
    """修改任务处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):
            try:
                data = event.data
                taskCode = data.get('code')
                sourceCode, stop = self.getSourceCode(data)
                if not sourceCode:
                    return Event(
                        event_type=EventType.RESULT,
                        data={'success': False, 'message': f"任务{taskCode}，未找到任务"},
                        source=self.name,
                        timestamp=time.time()
                    )
                if not self.taskisExist(taskCode):
                    result = False
                    message = f"任务{taskCode}，不存在"
                else:
                    result = self.scheduler.modifyTask(taskCode, sourceCode, data.get('cron'), stop)
                    message = f"任务{taskCode}，修改成功" if result else f"任务{taskCode}，修改失败"
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': result, 'message': message},
                    source=self.name,
                    timestamp=time.time()
                )
            except Exception as e:
                log.err("ModifyHandler", e)
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': False, 'message': str(e)},
                    source=self.name,
                    timestamp=time.time()
                )
    
    @property
    def supported_events(self):
        return ['task_modify']


class RemoveHandler(BaseTaskHandler):
    """删除任务处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):
            try:
                data = event.data
                taskCode = data.get('code')
                if not self.taskisExist(taskCode):
                    result = False
                    message = f"任务{taskCode}，不存在!"
                    self.update_status([taskCode], 0)
                else:
                    result = self.scheduler.removeTask(taskCode)
                    if result:
                        self.update_status([taskCode], 0)
                        log.warn(f"任务{taskCode}，删除成功，更新状态：{result}")
                        message = f"任务{taskCode}，删除成功"
                    else:
                        message = f"任务{taskCode}，删除失败"
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': result, 'message': message},
                    source=self.name,
                    timestamp=time.time()
                )
            except Exception as e:
                log.err("RemoveHandler", e)
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': False, 'message': str(e)},
                    source=self.name,
                    timestamp=time.time()
                )
    
    @property
    def supported_events(self):
        return ['task_remove', 'task_stop']


class GetNextRunTimeHandler(BaseTaskHandler):
    """获取下一次运行时间处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):
            try:
                data = event.data
                taskCode = data.get('code')
                message = ""
                if not self.taskisExist(taskCode):
                    result = False
                    message = f"任务{taskCode}，不存在!"
                else:
                    message = self.scheduler.getNextRun(taskCode)
                    if not message:
                        result = False
                        message = f"任务{taskCode}，查询下一次执行时间失败！"
                    else:
                        result = True
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': result, 'message': message},
                    source=self.name,
                    timestamp=time.time()
                )
            except Exception as e:
                log.err("GetNextRunTimeHandler", e)
                return Event(
                    event_type=EventType.RESULT,
                    data={'success': False, 'message': str(e)},
                    source=self.name,
                    timestamp=time.time()
                )
    
    @property
    def supported_events(self):
        return ['task_get_next_time']


class UnknownHandler(BaseTaskHandler):
    """未知命令处理类"""
    def handle(self, event: Event) -> Optional[Event]:
        message = f"未处理命令{event.event_type}"
        log.warn(f"未处理命令{event.event_type},{event.data.get('code')}")
        return Event(
            event_type=EventType.RESULT,
            data={'success': False, 'message': message},
            source=self.name,
            timestamp=time.time()
        )
    
    @property
    def supported_events(self):
        return ['*']  # 支持所有事件


class TaskManager(IWorker):
    """任务管理器"""
    def __init__(self, app: Sanic):
        self.app = app
        self.scheduler = Scheduler()
        self.bll = BaseBll(app=app)
        self.session = self.bll.session
    
    async def getData(self):
        """
        获取源码
        """
        try:
            from co6co_db_ext.db_utils import QueryListCallable
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
    
    def addTaskByCode(self, code: str, cron: str):
        task = ICustomTask.createInstance(code)
        if task:
            self.scheduler.addTask(code, task.main, cron, task.stop)
        else:
            log.warn("任务在代码中，未找到：{}".format(code))
        return task is not None
    
    def addTaskBySourceCode(self, code: str, sourceCode: str, cron: str):
        if self.scheduler.checkCode(sourceCode, cron):
            self.scheduler.addTask(code, sourceCode, cron)
            return True
        else:
            log.warn("检查代码失败：{}".format(code))
            return False
    
    def startTimeTask(self):
        """
        运行在数据库中的代码任务
        """
        taskArr = self.bll.run(self.getData)
        success = []
        faile = []
        for po in taskArr:
            code = po.get("code")
            category = po.get("category")  # 0 代码任务，1 表任务[代码在数据表中需要编译后才能运行]
            cron = po.get("cron")
            sourceCode = po.get("sourceCode")
            log.info("加载任务:{}}...".format(code))
            if category == 0:
                result = self.addTaskByCode(code, cron)
                success.append(code) if result else faile.append(code)
                continue
            # 任务在表中，已经关联表读取完成
            result = self.addTaskBySourceCode(code, sourceCode, cron)
            success.append(code) if result else faile.append(code)
        
        log.warn("加载任务完成,预加载：{}共加载,{}个任务".format(len(taskArr), self.scheduler.task_total))
        succ_result = 0
        fall_result = 0
        if len(success) > 0:
            print(*success)
            succ_result = self.bll.run(self.update_status, success, 1)
        if len(faile) > 0:
            fall_result = self.bll.run(self.update_status, faile, 0)
        exeStatue = self.bll.run(self.update_status__2)
        log.warn("状态更新,成功->{},失败->{},意外的状态：{}".format(succ_result, fall_result, exeStatue))
    
    def update_status(self, codeList: List[str] = None, status: int = 0) -> int:
        """
        更新状态
        codeList: 任务编码 None -->所有，[] --> 不更新，[,,,]--> 更新指定
        status: 状态 0: 停止 1:运行
        """
        try:
            if codeList and len(codeList) == 0:
                return 0
            if codeList is None:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val).values({SysTaskPO.execStatus: status})
            else:
                ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.enabled.val, SysTaskPO.code.in_(codeList)).values({SysTaskPO.execStatus: status})
            result = self.bll.run(db_tools.execSQL, self.session, ccc)
            self.bll.run(self.session.commit)
            return result
        except Exception as e:
            log.err("执行 ERROR", e)
            return None
    
    async def update_status__2(self):
        """
        意外状态更新
        """
        # 防止万一的代码
        ccc = Update(SysTaskPO).where(SysTaskPO.state == dict_state.disabled.val, SysTaskPO.execStatus == 1).values({SysTaskPO.execStatus: 0})
        result2 = await db_tools.execSQL(self.session, ccc)
        log.info("更新状态不正确的任务：{}【应该为0】".format(result2))
        await self.session.commit()
        return result2
    
    def start(self):
        """
        启动任务
        """
        self.startTimeTask()
    
    def stop(self):
        """
        停止任务
        """
        result = self.bll.run(self.update_status)
        self.scheduler.stop()
        log.warn("状态更新,成功->{}".format(result))
        log.info("等待其他任务退出..")
 