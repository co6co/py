import asyncio
from co6co.utils import log,json_util
from sqlalchemy import Select
from model.filters.DeviceFilterItems import posterTaskFilterItems
from utils.cvUtils import screenshot 
from sqlalchemy.ext.asyncio import AsyncSession
from model.pos.biz import bizCameraPO 
from sqlalchemy import or_, and_, Select
from co6co_db_ext.db_session import db_service
from co6co_db_ext.db_operations import DbOperations
import json,os
from typing import List
from co6co.utils.File import File 
import asyncio

class baseBll: 
    session:AsyncSession=None 
    loop=None
    def __init__(self,app,loop) -> None:
        _service:db_service=db_service(app.config.db_settings) 
        self.session:AsyncSession=_service.async_session_factory() 
        '''
        service:db_service=app.ctx.service
        self.session:AsyncSession=service.async_session_factory()
        '''
        self.loop=loop
        log.warn(f"..创建session。。")
        pass
    def __del__(self)->None:  
        log.info(f"{self}...关闭session")
        if self.session:asyncio.run(self.session.close())
        #if self.session: await self.session.close() 

    

    def __repr__(self) -> str:
        return f'{self.__class__}'
