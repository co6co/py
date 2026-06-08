from __future__ import annotations 
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from contextvars import ContextVar, Token

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session, sessionmaker, Session

# from model.pos.DbModelPo import User, Process

from sqlalchemy import select, create_engine, Engine
from sqlalchemy.orm import selectinload

import asyncio
import time
from typing import  Callable
from sqlalchemy.pool import NullPool
from co6co.utils import log 
from .po import BasePO
from . import auto_import_models 
from .appconfig import connectSetting

class db_service:
    default_settings= connectSetting.create_default() 
    session: scoped_session  # 同步连接 
    async_session_factory: sessionmaker  # 异步连接
    """
    AsyncSession 工厂函数
    sessionmaker 是个生成器类

    """
    useAsync: bool
    poolSize: int = None

    def createEngine(self, url, **kwargs) -> Engine:
        setting = {
            "ping": self.settings.get("pool_pre_ping"),
            "echo": True if isinstance(self.settings.get("echo"), bool) else self.settings.get("echo"),
            "pool_size": self.settings.get("pool_size"),
            "max_overflow": self.settings.get("max_overflow"),
            "poolclass": NullPool,
        }
        setting.update(kwargs)
        return create_engine(url, **setting)

    def createAsyncEngine(self, url, **kwargs):
        """
        创建异步引擎

        如果如果要使用新的 new_event_loop
        在调用该该方法的位置创建：
            custom_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(custom_loop)
        create_async_engine 会自动使用当前event_loop来创建
        """
        setting = {
            "pool_pre_ping": self.settings.get("pool_pre_ping"),
            "echo": True
            if isinstance(self.settings.get("echo"), bool)
            else self.settings.get("echo"),
            "pool_size": self.settings.get("pool_size"),
            "max_overflow": self.settings.get("max_overflow"),
        }
        setting.update(kwargs)
        return create_async_engine(url, **setting) #,

    def _session_factory(self, engine: Engine = None, **kv):
        if engine is None:
            engine = self.createEngine(self.url)
        default = {
            "autoflush": False,
            "autocommit": False,
        }
        default.update(kv)
        factory = sessionmaker(bind=self.engine, class_=Session, **default)
        return factory

    def _async_session_factory(
        self, engine: AsyncEngine = None, **kv
    ) -> Callable[[], AsyncSession]:
        """
        return AsyncSession 类
        """
        if engine is None:
            engine = self.createAsyncEngine(self.url)
        default = {
            "expire_on_commit": False,
        }
        default.update(kv)
        # #常见标注Type[AsyncSession]
        # 更准确  Callable[[], AsyncSession]
        factory = sessionmaker(engine, class_=AsyncSession, **default)
        return factory

    def createSession(self, engine: AsyncEngine = None, **kv):
        factory = self._session_factory(engine, **kv)
        return scoped_session(factory)

    def createAsyncSession(self, engine: AsyncEngine = None, **kv):
        factory = self._async_session_factory(engine, **kv)
        return factory()

    @property
    def Session(self)-> Callable[[], AsyncSession]:
        """
        创建AsyncSession 类
        """
        return self.async_session_factory

    def _createEngine(self, url: str):
        self.useAsync = True
        if "sqlite" in url:
            self.useAsync = False
            self.engine = self.createEngine(url)
            self.session = self.createSession(self.engine)
            BasePO.query = self.session.query_property()
        else:  # AsyncSession
            self.engine = self.createAsyncEngine(url)
            self.async_session_factory = self._async_session_factory(self.engine)

        self.base_model_session_ctx = ContextVar("session")

        pass
    @property
    def context_var(self):
        '''
        上下文变量
        使用方法
        token=context_var.set(session)  # 存入当前协程上下文
        session = context_var.get()  # 直接取出使用
        context_var.reset(token) #用户后重置防止全局污染 
        '''
        return self.base_model_session_ctx
    
     
    
    def __init__(self, config: connectSetting, engineUrl: str = None) -> None:
        """
        如果需使用新event_loop

        请使用
        async def main():
            current_loop = asyncio.get_event_loop()
            print(f"主函数中使用的事件循环: {current_loop} (ID: {id(current_loop)})")
            db=db_service(...)
            ....

        loop=create_event_loop()
        try:
            loop.run_until_complete(main())
        except:
            pass
        finally:
            loop.close()
        """

        self.settings = self.default_settings.copy()
        if engineUrl is None:
            self.settings.update(config)
            engineUrl = "mysql+aiomysql://{}:{}@{}/{}".format(
                self.settings["DB_USER"],
                self.settings["DB_PASSWORD"],
                self.settings["DB_HOST"],
                self.settings["DB_NAME"],
            )
        self.url = engineUrl
        self._createEngine(engineUrl)
        pass

    async def init_tables(self):
        if self.useAsync:
            async with self.engine.begin() as conn:
                # await conn.run_sync(BasePO.metadata.drop_a顶顶顶顶ll)
                await conn.run_sync(BasePO.metadata.create_all)
                await conn.commit()
            await self.engine.dispose()
        else:
            BasePO.metadata.create_all(bind=self.engine)

    def sync_init_tables(self,*models):
        auto_import_models(*models)
        retryTime = 0
        while True:
            try:
                if retryTime < 8:
                    retryTime += 1
                asyncio.run(self.init_tables())
                break
            except Exception as e:
                log.warn(f"同步数据表失败{e}!")
                log.info(f"{retryTime * 5}s后重试...")
                time.sleep(retryTime * 5)
