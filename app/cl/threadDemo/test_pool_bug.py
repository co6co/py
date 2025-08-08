from __future__ import annotations
from re import A
from co6co.utils.network import ping_host
from co6co_db_ext.po import BasePO
from sqlalchemy import Integer , Column,  String, DateTime
from concurrent.futures import Future
from co6co.task.pools import   limitThreadPoolExecutor as ThreadPoolExecutor,ThreadPool
from co6co_db_ext.session import dbBll 

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from co6co_db_ext.db_utils import db_tools 
from co6co.utils import log 

class DemoPO(BasePO): 
    __tablename__ = "demo"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="名称") 
    code = Column("code", String(64),  comment="编码")
    age= Column("age", Integer,  comment="代码编码")
    updateTime = Column("update_time", DateTime, comment="更新时间")
    createTime = Column("create_time", DateTime, comment="创建时间") 
async def update(session:AsyncSession, name,code,age):
    """
    插入更新表
    @param name 名称
    @param code 编码
    @param age 年龄
    @return 更新?,影响行数 
    """
    try: 
        log.warn("kakakak....")
        exist=await db_tools.exist(session,DemoPO.code.__eq__(code),column=DemoPO.id)
        if exist:
            log.info("kakakak.过.SQL1")
            result =await db_tools.execSQL(session,text("update demo set name=:name,age=:age where code=:code"),{"name":name,"code":code,"age":age})  
        else: 
            result=await db_tools.execSQL(session,text("insert into demo (name,code,age) values (:name,:code,:age)"),{"name":name,"code":code,"age":age})  
        await session.commit()    
        return exist,result
    except Exception as e:
        print("error,update,",e)
def createDB():
    bll=dbBll(db_settings={"DB_HOST":"localhost","DB_PORT":3306,"DB_USER":"root","DB_PASSWORD":"mysql123456","DB_NAME":"test","echo":False})
    return bll 
def updateDb(name,code,pingResult):
    bll=createDB()
    try: 
        result=bll.run(update,bll.session,name,code,18 if pingResult else 28) 
        return result 
    except Exception as e:
        print("error,updateDb,",e)
    finally:
        bll.close()   

def check(ip:str): 
    return ping_host(ip)
def check2(name,code):
    import time
    time.sleep(5)
    return True,1
    updated0, result0=updateDb(name,code,result)  
    log.succ(f"{name}{'更新' if updated0 else '插入'} 影响行数：{result0}")
    return updated0, result0
def result(f: Future): 
    if f.exception(): 
        print(f"执行任务发生错误: {f.exception()}")
    else: 
        code=f.code
        name=f.name
        result = f.result()  
        log.info(f"ping{name} {result},准备更新数据库...") 
        log.info(f"ping{name} {result},准备更新数据库...") 
        updated0, result0=updateDb(name,code,result)  
        log.succ(f"{name}{'更新' if updated0 else '插入'} 影响行数：{result0}")
def result2(f: Future):
    if f.exception(): 
        print(f"执行任务发生错误: {f.exception()}")
    else:  
        result = f.result()  
        log.succ(result)

 
    
def poolWork(item:dict):
    name=item.get("name")
    code=item.get("code") 
    log.info("index->",item.get("id"))
    result=ping_host(name,1,1)
    updated0, result0=updateDb(name,code,result)
    log.succ(f"{'更新' if updated0 else '插入'} 影响行数：{result0}")  
    
def main():
    bll=createDB() 
    list=bll.run(db_tools.execForMappings,bll.session,text("select * from demo")) 
    bll.close() 
    #
    # //todo  需要解决 limitThreadPoolExecutor 使用 bllDb + ping 卡死问题
    #
    with ThreadPoolExecutor(2, "thread_pool") as executor :   
        
        #results=list(executor.map(check2,list))
        #print("所有结果：", results)

        for item in list:   
            item:dict=item 
            name=item.get("name")
            code=item.get("code") 
            print("index->",item.get("id"))
            #future0=executor.submit(check,name )
            future0=executor.submit(check2,name,code )
            future0.code=code
            future0.name=name
            #future0.add_done_callback(result2)
            updated0,result0=future0.result() 
            print(f"{"更新" if updated0 else "插入"} 影响行数：",result0) 
    #pool=ThreadPool(max_workers=20)
    #for item in list:   
    #    item:dict=item  
    #    #pool.submit(lambda:poolWork(item))
    #    pool.submit(lambda n=item: poolWork(n)) 
    #log.warn("等等所有线程结束..")
    #pool.join()
    #log.warn("所有线程结束")
 
if __name__ == '__main__':
    main()
    print('done')
