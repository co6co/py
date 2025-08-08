from __future__ import annotations
from co6co_db_ext.po import BasePO
from sqlalchemy import column, func, INTEGER, DATE, FLOAT, DOUBLE, SMALLINT, Integer, UUID, Text, INTEGER, BigInteger, Column, ForeignKey, String, DateTime



from co6co.task.pools import  limitThreadPoolExecutor as ThreadPoolExecutor
from co6co_db_ext.session import dbBll 
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from co6co_db_ext.db_utils import db_tools 
class DemoPO(BasePO): 
    __tablename__ = "demo2"
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
        return exist,result
    except Exception as e:
        print("error,update,",e)

def updateDb(name,code,age):
    bll=dbBll(db_settings={"DB_HOST":"localhost","DB_PORT":3306,"DB_USER":"root","DB_PASSWORD":"mysql123456","DB_NAME":"test","echo":True})
    try:
        result=bll.run(update,bll.session,name,code,age) 
        return result 
    except Exception as e:
        print("error,updateDb,",e)
    finally:
        bll.close()


def main():
    with ThreadPoolExecutor(1, "插入更新数据") as executor:
        for i in range(650,680,3): 
            print("index->",i)
            future0=executor.submit(updateDb,"test"+str(i),"123_"+str(i),18+i)
            updated0,result0=future0.result() 
            print(f"{"更新" if updated0 else "插入"} 影响行数：",result0) 
            future1=executor.submit(updateDb,"test"+str(i+1),"123_"+str(i+1),18+i+1)
            updated1,result1=future1.result() 
            print(f"{"更新" if updated1 else "插入"} 影响行数：",result1) 
            future2=executor.submit(updateDb,"test"+str(i+2),"123_"+str(i+2),18+i+2) 
            updated2,result2=future2.result()  
            print(f"{"更新" if updated2 else "插入"} 影响行数：",result2) 

if __name__ == '__main__':
    main()
    print('done')
