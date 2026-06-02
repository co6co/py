from co6co.utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from .db_session import db_service, connectSetting
from co6co.task.thread import ThreadEvent
import functools
from contextlib import asynccontextmanager
  

class session_context:
    """
    会话上下文管理器
    async with SessionContext(session)() as session: 
        ....
    """ 
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @asynccontextmanager
    async def __call__(self):
        """使实例可直接作为上下文管理器使用"""
        async with self.session as session, session.begin():
            yield session
def transactional(func):
    """
    事务装饰器

    示例：
    @transactional
    async def your_method(session,select):
        result = await session.execute(select)
        ...
        return result
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        _session:AsyncSession = self.session if hasattr(self, "session")   else self.db_session
        if _session is None:
            raise Exception("session,参数不能为空")
        async with _session as session, session.begin():
            return await func(self, *args, **kwargs)
    return wrapper
class dbBll:
    def __init__(self, *,  db_settings: connectSetting = {}) -> None:
        self.t = ThreadEvent()
        if not db_settings:
            raise Exception("db_settings,参数不能为空")

        self.db_settings = db_settings
        self.session = None
        self.service = None
        self.t.runTask(self.create_db)
        self.closed = False

    async def create_db(self):
        # current_loop = asyncio.get_event_loop()
        # print(f"主函数中使用的事件循环: {current_loop} (ID: {id(current_loop)}),{id(self.t.loop)}")
        _service: db_service = db_service(self.db_settings)
        self.session: AsyncSession = _service.async_session_factory()
        self.service = _service

    def run(self, task, *args, **argkv):
        data = self.t.runTask(task, *args, **argkv)
        return data

    def close(self):
        self.closed = True
        self.t.runTask(self.session.close)
        self.t.runTask(self.service.engine.dispose)
        self.t.close()

    def __str__(self):
        return f'{self.__class__}'

    def __del__(self) -> None:
        try:
            if not self.closed:
                self.t.run(self.close)

        except Exception as e:
            log.warn("__del___ error", e)
            pass
