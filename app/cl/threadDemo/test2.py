from __future__ import annotations
from co6co_db_ext.po import BasePO
from sqlalchemy import column, func, INTEGER, DATE, FLOAT, DOUBLE, SMALLINT, Integer, UUID, Text, INTEGER, BigInteger, Column, ForeignKey, String, DateTime

from concurrent.futures import Future

from co6co.task.pools import  limitThreadPoolExecutor as ThreadPoolExecutor
from co6co_db_ext.session import dbBll 
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from co6co_db_ext.db_utils import db_tools 
import asyncio
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
        exist=await db_tools.exist(session,DemoPO.code.__eq__(code),column=DemoPO.id)
        if exist:
            result =await db_tools.execSQL(session,text("update demo set name=:name,age=:age where code=:code"),{"name":name,"code":code,"age":age})  
        else:
            result=await db_tools.execSQL(session,text("insert into demo (name,code,age) values (:name,:code,:age)"),{"name":name,"code":code,"age":age})  
        await session.commit()    
        await asyncio.sleep(5)
        return exist,result
    except Exception as e:
        print("error,update,",e)
def createDB():
    bll=dbBll(db_settings={"DB_HOST":"localhost","DB_PORT":3306,"DB_USER":"root","DB_PASSWORD":"mysql123456","DB_NAME":"test","echo":True})
    return bll

def updateDb(name,code,age):
    bll=createDB()
    try:
        result=bll.run(update,bll.session,name,code,age) 
        return result 
    except Exception as e:
        print("error,updateDb,",e)
    finally:
        bll.close() 
def result(f: Future): 
    if f.exception(): 
        print(f"执行任务发生错误: {f.exception()}")
    else: 
        updated0, result0 = f.result()  
        print(f"{'更新' if updated0 else '插入'} 影响行数：{result0}")
def bck(f: Future ): 
    # 执行完毕后 ，才会执行 result方法
    f.add_done_callback(result)

def main():
    bll=createDB() 
    list=bll.run(db_tools.execForMappings,bll.session,text("select * from demo"))
    print(list) 
    bll.close()
    print(list) 
    with ThreadPoolExecutor(4, "插入更新数据") as executor:
        for item in list: 
            item:dict=item
            name=item.get("name")
            code=item.get("code")
            age=item.get("age")

            print("index->",item.get("id"))
            future0=executor.submit(updateDb,name,code,age)
            bck(future0)  
            break
            #updated0,result0=future0.result() 
            #print(f"{"更新" if updated0 else "插入"} 影响行数：",result0)  
if __name__ == '__main__':
    main()
    print('done')
