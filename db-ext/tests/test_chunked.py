
from sqlalchemy.sql import text

from .right import UserPO
from sqlalchemy import Select,Delete,Update
from co6co_db_ext.actuator import Actuator
from co6co.utils import log
from sqlalchemy import func
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.engine.cursor import CursorResult 
from sqlalchemy.ext.asyncio.result import AsyncResult
async def test_select_entity(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    stmt = Select(UserPO)
    result=await actuator2._execute(stmt) 
    log.warn("select entity result",  type(result))
    assert isinstance(result, ChunkedIteratorResult)

async def test_select_column(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    stmt = Select(UserPO.id)
    result=await actuator2._execute(stmt) 
    log.warn("select column result",  type(result))
    assert isinstance(result, ChunkedIteratorResult)

async def test_select_func(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    stmt = Select(func.count(UserPO.id)).where(UserPO.id > 0)
    result=await actuator2._execute(stmt) 
    log.warn("select func result",  type(result))
    assert isinstance(result, ChunkedIteratorResult)

async def test_delete_func(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
  
    stmt = Delete(UserPO).where(UserPO.id == 0)
    result=await actuator2._execute(stmt) 
    log.warn(" delete result",  type(result))
    assert isinstance(result, CursorResult)

async def test_update_func(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
  
    stmt = Update(UserPO).where(UserPO.id == 0).values({UserPO.userName:"test"})
    result=await actuator2._execute(stmt) 
    log.warn("update result",  type(result))
    assert isinstance(result, CursorResult)

async def test_text_func(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
  
    stmt = text("select id from sys_user")
    result=await actuator2._execute(stmt) 
    log.warn("text_select result",  type(result))
    assert isinstance(result, CursorResult)

async def test_text_stream_func(db_service_param):
    """
    百万级数据导出
    需要使用 流式
    """
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    stmt = text("select id,user_name from sys_user").execution_options(stream_results=True)
    #result=await actuator2.session.stream(stmt) 
    #rows = result.all()   # ✅ 必须读完 否则触发 Previous unbuffered result was left incomplete warnings.warn
    #print("rows", rows[0]) 
    result = await actuator2.session.stream(stmt)   
    async  for row in result:  #AsyncResult并不是一个异步上下文管理器
        print("row->id", row._mapping["id"], "row->userName", row._mapping["user_name"])
        if row._mapping["id"] > 100:
            break
    log.warn("text_stream result",  type(result))
    await result.close()
    assert isinstance(result, AsyncResult)

async def test_text_select_func(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
  
    stmt = text("select 1")
    result=await actuator2._execute(stmt) 
    log.warn("text_select result",  type(result))
    assert isinstance(result, CursorResult)
