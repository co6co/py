# 优化版本：使用真正的异步数据库驱动和异步 ping
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import aiohttp  # 用于异步 HTTP 请求

# 异步 ping 实现
async def async_ping(ip: str) -> bool:
    try:
        # 使用 aiohttp 发送异步 HTTP 请求或使用其他异步网络库
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{ip}", timeout=1) as response:
                return response.status == 200
    except:
        return False

# 异步数据库操作
async def async_db_operation(name: str, code: str, age: int):
    # 创建异步数据库引擎
    engine = create_async_engine(
        "mysql+aiomysql://root:mysql123456@localhost:3306/test",
        echo=True
    )
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session=async_session()
    async with session,session.begin(): 
            # 执行异步数据库操作
            # ...
            return True, 1  # 模拟结果

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
    for i in range(1, 11):
        ip = f"192.168.1.{i}"
        name = f"test{i}"
        code = f"code{i}"

        # 异步 ping
        ping_result = await async_ping(ip)
        age = 18 if ping_result else 28

        # 异步数据库操作
        tasks.append(async_db_operation(name, code, age))

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    # 处理结果...

if __name__ == '__main__':
    asyncio.run(main())