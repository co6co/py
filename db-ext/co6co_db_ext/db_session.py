from sqlalchemy.ext.asyncio import create_async_engine

from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session,sessionmaker

#from model.pos.DbModelPo import User, Process

from sqlalchemy import select
from sqlalchemy.orm import selectinload 
 
import asyncio,time ,typing 
from typing import TypeVar
from co6co.utils import log
from co6co_db_ext.po import BasePO

class db_service:
    session:scoped_session  # 同步连接
    async_session_factory:sessionmaker #异步连接
    useAsync:bool
    def _createEngine(self, url:str ):
        self.useAsync=True
        if "sqlite" not in  url: 
            self.engine = create_async_engine(url, echo=True )  
            self.async_session_factory  = sessionmaker(self.engine, expire_on_commit=False,class_=AsyncSession)# AsyncSession,
        else:
            self.useAsync=False
            self.session=scoped_session( sessionmaker(autoflush=False, autocommit=False,bind=self.engine) ) 
            BasePO.query=self.session.query_property()
        self.base_model_session_ctx = ContextVar("session") 
        pass
        
    def __init__(self,settings:dict,engineUrl:str=None  ) -> None:  
        defaultSetting={
            'DB_HOST': 'localhost',
            'DB_NAME': '',
            'DB_USER': 'root',
            'DB_PASSWORD':''
        }
        self.settings=defaultSetting.update(settings)
        if  engineUrl!=None:self._createEngine(engineUrl)
        else :
            engineUrl=f"mysql+aiomysql://{defaultSetting['DB_USER']}:{defaultSetting['DB_PASSWORD']}@{defaultSetting['DB_HOST']}/{defaultSetting['DB_NAME']}"
            self._createEngine(engineUrl ); 
        pass
    async def init_tables(self):
        async with self.engine.begin() as conn: 
            #await conn.run_sync(BasePO.metadata.drop_all)
            await conn.run_sync(BasePO.metadata.create_all) 
            await conn.commit() 
        await self.engine.dispose() 
  

    def sync_init_tables(self): 
         retryTime=0
         while(True):
            try:
                if retryTime<8:retryTime+=1
                asyncio.run(self.init_tables())
                break
            except Exception as e:
                log.err(f"创建数据表失败{e}") 
                log.info(f"{retryTime*5}s后重试...")
                time.sleep(retryTime*5) 