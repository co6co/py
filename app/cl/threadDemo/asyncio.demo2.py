# 优化版本：使用真正的异步数据库驱动和异步 ping
import asyncio
import time
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext.session import dbBll
from sqlalchemy.orm import session
from co6co_db_ext.db_utils import db_tools
from co6co.utils.network import ping_host
from sqlalchemy.sql import text
from sqlalchemy import Integer, Column,  String, DateTime
from co6co .utils import log
from co6co_db_ext.po import BasePO
from co6co.task.pools import run_async_tasks
import asyncio


class DemoPO(BasePO):
    __tablename__ = "demo2"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="名称")
    ip = Column("ip", String(64),  comment="ip")
    age = Column("age", Integer,  comment="代码编码")
    updateTime = Column("update_time", DateTime, comment="更新时间")
    createTime = Column("create_time", DateTime, comment="创建时间")
# 异步 ping 实现


async def async_ping(ip: str) -> bool:
    return ping_host(ip)

# lock = threading.Lock()  # 创建锁对象
lock = asyncio.Lock()  # 创建锁对象
# 异步的 update 函数（与原代码相同）


async def update(session: AsyncSession, name, ip, age):
    """
    存在的问题：
    1. 不能使用 session
        This session is provisioning a new connection; concurrent operations are not permitted
    2. 不能使用session, with lock  卡死

    """
    try:
        # bll=createDb()
        # session=bll.session
        log.warn("操作 ", ip, '...')
        async with lock:
            log.warn("进入锁..ed")
            exist = await db_tools.exist(session, DemoPO.ip.__eq__(ip), column=DemoPO.id)
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
            print("11")
            await session.commit()
            return exist, result

    except Exception as e:
        print("error,update,", e)
        return False, 0
    finally:
        # bll.close()
        log.warn("退出锁..ed")


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
# 主函数
# 这种方式完全避免了线程池，使用真正的异步 IO 操作，性能和可靠性会更好。
# 但需要确保所有依赖库都支持异步操作。


async def main():
    """
    主函数
    性能和可靠性会更好

    1. 完全避免了线程池
    2. 真正的异步 IO 操
    3. 确保所有依赖库都支持异步操作
    """
    tasks = []
    bll = createDb()
    for i in range(1, 5):
        ip = f"192.168.1.{i}"
        name = f"test{i}"
        # 异步 ping
        print("ping", ip, '...')
        ping_result = await async_ping(ip)
        print("ping", ip, ping_result)
        age = 18 if ping_result else 28
        # 异步数据库操作
        tasks.append(update(bll.session, name, ip, age))

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    # 处理结果
    for i, (updated, result) in enumerate(results):
        log.succ(f"任务 {i+1}: {'更新' if updated else '插入'} 影响行数: {result}")
    # 处理结果...
    print("关闭session..")

    bll.close()
    print("关闭session....")


async def tempTask(item: dict):
    session, name, ip, age = item.values()
    ping_result = await async_ping(ip)
    print("ping", ip, ping_result)
    age = 18 if ping_result else 28
    return await update(session, name, ip, age)


async def main2():
    bll = createDb()
    try:
        params = [{"session": bll.session, "name": f"test{i}", "ip": f"192.168.1.{i}", "age": 18} for i in range(1, 5)]
        results = await run_async_tasks(params, tempTask)
        for i, (updated, result) in enumerate(results):
            log.succ(f"任务 {i+1}: {'更新' if updated else '插入'} 影响行数: {result}")
    finally:
        bll.close()
if __name__ == '__main__':
    # asyncio.run() 会自动创建新的事件循环，并在函数结束后销毁，因此在其外部创建的锁会失效。
    asyncio.run(main())
    # main 和 main2 是等效的
    print(" 开始执行main2...")

    # 锁被  <asyncio.locks.Lock object at 0x0000025C4DCBF260 [locked]> is bound to a different event loop
    # 需要 重新创建
    lock = asyncio.Lock()
    asyncio.run(main2())
