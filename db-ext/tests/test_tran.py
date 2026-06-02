from co6co_db_ext.db_utils import DbCallable
from co6co_db_ext.actuator import Actuator
from co6co_db_ext.session import session_context
from co6co_db_ext.db_utils import Select
from sqlalchemy import text


async def test_start_two_transactions(db_service_param):
    _, _, actuator = db_service_param
    actuator: Actuator = actuator
    # 为什么 第二个Session 没有关闭
    async with session_context(actuator.session)() as session:
        print(await actuator.execute(text("select 1,2")))
        print(session.is_active, id(session))
        await session.commit()
        await session.close()
    async with session_context(actuator.session)() as session:
        print(await actuator.execute(text("select 2,3")))
        print(session.is_active, id(session))
        await session.commit()
        await session.close()


async def test_start_two_transactions2(db_service_param):
    _, factory, actuator = db_service_param
    sesstion = factory()
    # 为什么 第二个Session 没有关闭
    async with session_context(sesstion)() as session:
        print(await actuator.execute(text("select 1,2")))
        print(session.is_active, id(session))
        await session.commit()
        await session.close()
    async with session_context(sesstion)() as session:
        print(await actuator.execute(text("select 2,3")))
        print(session.is_active, id(session))
        await session.commit()
