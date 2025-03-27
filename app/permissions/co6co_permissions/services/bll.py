import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext.db_session import db_service
import asyncio
from sanic import Sanic
from co6co.task.thread import ThreadEvent


class baseBll:
    session: AsyncSession = None

    def __init__(self, db_settings: dict) -> None:
        _service: db_service = db_service(db_settings)
        self.session: AsyncSession = _service.async_session_factory()
        '''
        service:db_service=app.ctx.service
        self.session:AsyncSession=service.async_session_factory()
        '''
        # log.warn(f"..创建session。。")
        pass

    def __del__(self) -> None:
        # log.info(f"{self}...关闭session")
        if self.session:
            asyncio.run(self.session.close())
        # if self.session: await self.session.close()

    def __repr__(self) -> str:
        return f'{self.__class__}'


class BaseBll(baseBll):
    t: ThreadEvent

    def __init__(self,*,  db_settings: dict={},app:Sanic=None) -> None:  
        self.t = ThreadEvent() 
        if not db_settings:
            app =app or Sanic.get_app()
            db_settings=app.config.db_settings 
        super().__init__(db_settings)

    def run(self, task, *args, **argkv):
        data = self.t.runTask(task, *args, **argkv)
        return data

    def __del__(self) -> None:
        try:
            log.info(f"{self}...关闭session") 
            if self.session:
                self. t.runTask(self.session.close)
            # if self.session: await self.session.close()
        except Exception as e:
            log.warn("__del___ error",e)
            pass

