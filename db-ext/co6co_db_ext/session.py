import asyncio
from co6co.utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext.db_session import db_service
from typing import List
import asyncio
from sanic import Sanic
from co6co.task.thread import ThreadEvent


class DbSession:
    """
    创建 Session

    """
    session: AsyncSession = None

    def __init__(self, db_settings: dict) -> None:
        _service: db_service = db_service(db_settings)
        self.session: AsyncSession = _service.async_session_factory()
        '''
        service:db_service=app.ctx.service
        self.session:AsyncSession=service.async_session_factory()
        '''
        log.warn(f"..创建session。。")
        pass

    def __del__(self) -> None:
        if self.session:
            asyncio.run(self.session.close())
            # log.info(f"{self}...关闭session")

    def __repr__(self) -> str:
        return f'{self.__class__}'


class BaseBll(DbSession):
    """
    数据库操作
    定义异步方法
    运行 result=run(异步方法,arg)

    """

    _eventLoop: ThreadEvent

    def __init__(self, db_settings: dict) -> None:
        self._eventLoop = ThreadEvent()
        super().__init__(db_settings)

    def run(self, task, *args, **argkv):
        data = self._eventLoop.runTask(task, *args, **argkv)
        return data

    def __del__(self) -> None:
        if self.session:
            self._eventLoop.run(self.session.close())
        self._eventLoop.close()
        # if self.session: await self.session.close()
