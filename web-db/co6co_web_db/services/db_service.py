from sqlalchemy.ext.asyncio import create_async_engine

from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

#from model.pos.DbModelPo import User, Process

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sanic.response import json

from sanic import Sanic,Blueprint 
from sanic.blueprint_group import BlueprintGroup
import asyncio,time ,typing
from sanic.log import logger  
from sanic import Blueprint,Request
from typing import TypeVar
from co6co_db_ext.db_session import db_service

'''class db_service: 
    def __init__(self,app:Sanic,settings:dict,basePoType:TypeVar) -> None:
        self.app=app
        self.basePO=basePoType

        defaultSetting={
            'DB_HOST': 'localhost',
            'DB_NAME': '',
            'DB_USER': 'root',
            'DB_PASSWORD':''
        }
        self.settings=defaultSetting.update(settings)
        self.engine = create_async_engine(f"mysql+aiomysql://{defaultSetting['DB_USER']}:{defaultSetting['DB_PASSWORD']}@{defaultSetting['DB_HOST']}/{defaultSetting['DB_NAME']}", echo=True) 
        self.async_session_factory  = sessionmaker(self.engine, expire_on_commit=False,class_=AsyncSession)# AsyncSession,
        self.base_model_session_ctx = ContextVar("session") 
        pass
    async def init_tables(self):
        async with self.engine.begin() as conn: 
            #await conn.run_sync(BasePO.metadata.drop_all)
            await conn.run_sync(self.basePO.metadata.create_all) 
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
                logger.error(e)
                logger.info(f"{retryTime*5}s后重试...")
                time.sleep(retryTime*5) '''
                
def injectDbSessionFactory(app:Sanic,settings:dict,engineUrl:str=None ):
    """
    挂在 DBSession_factory
    """
    service=db_service(settings,engineUrl )
    service.sync_init_tables() 
    '''
    @app.main_process_start
    async def inject_session_factory(app:Sanic):
        logger.info("挂在db_session_factory。。。")  
        app.shared_ctx.cache=multiprocessing.Manager().dict()
        app.shared_ctx.cache["db_session_factory"]=_async_session_factory 

    ''' 
    @app.middleware("request")
    async def inject_session(request:Request): 
        if "/api" in request.path:
            #logger.info("mount DbSession 。。。")
            if service.useAsync:
                request.app.ctx.session_fatcory=service.async_session_factory
                request.ctx.session=service.async_session_factory() 
                request.ctx.session_ctx_token = service.base_model_session_ctx.set(request.ctx.session) 
            else:
                request.ctx.session=service.session 
        
    @app.middleware("response")
    async def close_session(request:Request, response):
        if hasattr(request.ctx, "session_ctx_token"):
            try:
                if service.useAsync:
                    service.base_model_session_ctx.reset(request.ctx.session_ctx_token) 
                    await request.ctx.session.close()
                #logger.info("close DbSession。")
                #await request.ctx.session.dispose() 
            except Exception as e:
                logger.err(e) 
                #logger.error("close DbSession。Error")
 

    
