import pytest
from co6co_db_ext.db_session import db_service

from co6co_db_ext.actuator import Actuator
import time
from sqlalchemy.sql import text
from co6co.utils import log

async def exe_sql(session):
    actuator = Actuator(session)
    result = await actuator._execute(
        text("SHOW VARIABLES LIKE 'max_connections'")
    )
    max_connections = int(result.all()[0][1])
    print("最大连接数：", max_connections)
    result = await actuator._execute(
        text("SHOW STATUS LIKE 'Threads_connected';")
    )
    threads_connected =int(result.all()[0][1])
    print("当前连接数：", threads_connected)
    assert  threads_connected <= max_connections
    return threads_connected

def assert_connect_num_increase(current:list[int]):
    log.warn("连接数是否增加",current)
    for i,_ in enumerate(current): 
        if i==0:
            continue
        result=current[i-1]>=current[i]
        if result:
            break
    
    assert result,f"连接数一直在增加->{current}"  
async def exec_connect_num(db_service_param2,times=10):
    service: db_service = db_service_param2
    current=[]
    for i in range(times):
        async with service.Session() as session:
            current.append(await exe_sql(session))
        time.sleep(0.5) 
    assert_connect_num_increase(current) 


async def exec_connect2_num(db_service_param2,times=10):
    service: db_service = db_service_param2
    current=[] 
    for i in range(times):
        try:
            session=service.Session()
            await session.begin()
            current.append(await exe_sql(session))
            await session.commit() 
        except Exception as e:
            await session.rollback() 
            raise e
        finally:
            #session.close()
            pass
        time.sleep(0.5) 
    assert_connect_num_increase(current) 

async def test_current_connect(db_service_param2):
    service: db_service = db_service_param2
    sql = """
        SELECT
            SUBSTRING_INDEX(HOST, ':', 1) AS client_ip,
            COUNT(*) AS connection_count
        FROM
            information_schema.PROCESSLIST
        GROUP BY
            client_ip
        ORDER BY
            connection_count DESC
        LIMIT 10;
    """
    select = text(sql)
    await exec_connect_num(db_service_param2) 
    async with service.Session() as session:
        actuator = Actuator(session)
        result = await actuator.query_one_mappings(select)
        print(result)
        
async def test_current_connect_by_begin(db_service_param2): 
    await exec_connect2_num(db_service_param2,30)
     
async def test_current_connect_by_one_session(db_service_param2): 
    service: db_service = db_service_param2
    current=[] 
    session=service.Session()
    for i in range(10):
        try: 
            await session.begin()
            current.append(await exe_sql(session))
            await session.commit() 
        except Exception as e:
            await session.rollback() 
            raise e
        finally:
            #session.close()
            pass
        time.sleep(0.5) 
    assert_connect_num_increase(current) 