import asyncio
from operator import eq
import time
from concurrent.futures import ThreadPoolExecutor
from co6co.task.pools import   limitThreadPoolExecutor as ThreadPoolExecutor #,ThreadPool
from co6co.utils.network import ping_host
from co6co_db_ext.session import dbBll
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from db_utils import db_tools
from co6co.utils import log
from co6co_db_ext.po import BasePO
from sqlalchemy import Integer , Column,  String, DateTime, column
class DemoPO(BasePO): 
    __tablename__ = "demo2"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="名称") 
    ip = Column("ip", String(64),  comment="ip")
    age= Column("age", Integer,  comment="代码编码")
    updateTime = Column("update_time", DateTime, comment="更新时间")
    createTime = Column("create_time", DateTime, comment="创建时间") 
# 模拟阻塞的 ping 操作
def blocking_ping(ip: str) -> bool: 
    return ping_host(ip) # 模拟阻塞 有网络IO

# 异步的 update 函数（与原代码相同）
async def update(session: AsyncSession, name, ip, age):
    try:
        exist = await db_tools.exist(session, DemoPO.ip.__eq__(ip),column=DemoPO.id)
        if exist:
            result = await db_tools.execSQL(
                session, 
                text("update demo2 set name=:name,age=:age where ip=:ip"),
                {"name": name, "ip": ip, "age": age}
            )
        else:
            result = await db_tools.execSQL(
                session, 
                text("insert into demo2 (name,ip,age) values (:name,:ip,:age)"),
                {"name": name, "ip": ip, "age": age}
            )
             
        await session.commit()
        return exist, result
    except Exception as e:
        print("error,update,", e)
        return False, 0
def createDb():
    bll = dbBll(db_settings={
        "DB_HOST": "localhost",
        "DB_PORT": 3306,
        "DB_USER": "root",
        "DB_PASSWORD": "mysql123456",
        "DB_NAME": "test",
        "echo": True
    })
    return bll
# 模拟阻塞的数据库操作
def upload_sync(name: str, ip: str, age: int) -> tuple:
    # 创建数据库连接
    bll=createDb()
    try:
        # 使用 run 方法执行异步数据库操作
        result = bll.run(update, bll.session, name, ip, age)
        return result
    finally:
        bll.close()
def checkAll(name: str, ip: str, age: int):
    ping_result=blocking_ping(ip)
    db_age = age if ping_result else 28 
    db_task= upload_sync(name, ip, db_age)
    return db_task

# 异步主函数
async def main():
    bll=createDb()
    await bll.service.init_tables()
    bll.close()

    # 创建线程池
    with ThreadPoolExecutor(max_workers=1) as executor: 
        tasks = [] 
        # 创建多个任务
        for i in range(1, 10):
            ip = f"192.168.1.{i}"
            name = f"test{i}" 
            age = 18 + i
            print("ip->",ip ,"...") 
            db_task=executor.async_task(checkAll,name, ip, age) 
            tasks.append(db_task)

        # 等待所有数据库操作完成
        results = await asyncio.gather(*tasks)

        # 处理结果
        for i, (updated, result) in enumerate(results):
            log.succ(f"任务 {i+1}: {'更新' if updated else '插入'} 影响行数: {result}")

if __name__ == '__main__':
    asyncio.run(main())