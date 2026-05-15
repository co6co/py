from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session, sessionmaker, Session

# from model.pos.DbModelPo import User, Process

from sqlalchemy import select, create_engine, Engine
from sqlalchemy.orm import selectinload

import asyncio
import time
from typing import TypeVar, TypedDict, Type, Callable
from sqlalchemy.pool import NullPool
from co6co.utils import log
from co6co.task.thread import ThreadEvent
from co6co.data import DictNamespace
from .po import BasePO
from . import auto_import_models

class connectSetting(TypedDict):
    DB_HOST: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    echo: bool
    pool_size: int
    max_overflow: int
    pool_pre_ping: bool
    # 不支持示例方法
    #def from_(self, data: DictNamespace):
    #    for k in self.keys():
    #        self[k] = data.get(k)
    @staticmethod
    def from_(instance:TypedDict, data: DictNamespace):
        for k in instance.keys():
            instance[k] = data.get(k)
    @classmethod
    def create_default(cls, data: DictNamespace = None):
        instance = cls(
            DB_HOST="localhost",
            DB_NAME="",
            DB_USER="root",
            DB_PASSWORD="",
            echo=True,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # 执行sql语句前悲观地检查db是否可用
            # 'pool_recycle':1800 #超时时间 单位s
        )
        if data is not None:
            connectSetting.from_(instance, data)
        return instance


class db_service:
    default_settings: connectSetting = connectSetting.create_default()
    settings = {}
    session: scoped_session  # 同步连接

    async_session_factory: sessionmaker  # 异步连接
    """
    AsyncSession 工厂函数
    sessionmaker 是个生成器类

    """
    useAsync: bool
    poolSize: int = None
    poolSize: int = None

    def createEngine(self, url, **kwargs) -> Engine:
        setting = {
            "ping": self.settings.get("pool_pre_ping"),
            "echo": True
            if isinstance(self.settings.get("echo"), bool)
            else self.settings.get("echo"),
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
        return create_async_engine(url, **setting)

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
    def Session(self):
        """
        创建AsyncSession 类
        """
        return self._async_session_factory()

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
