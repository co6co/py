from sqlalchemy.sql import Select

from co6co_db_ext.actuator import Actuator
import pytest
from co6co_db_ext.db_session import db_service, connectSetting
from .right import *


def get_actuator(db_service_param):
    cfg, Session, actuator = db_service_param
    return actuator


async def test_actuator(db_service_param):
    cfg, Session, actuator = db_service_param
    print("test_actuator", cfg)
    pass


async def test_actuator_add_all(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    print("test_actuator_add_all", cfg)
    po = UserPO()
    po.userName = "test_add"
    stmt = Select(UserPO).where(UserPO.userName == po.userName)
    result = await actuator2.query_one_entity(stmt)
    if result is not None:
        await actuator2.delete(result)
        await actuator2.session.commit()
    actuator2.add_all(*[po])
    await actuator2.session.commit()
    await actuator2.session.close_all()
    pass


async def add_user_group(actuator2: Actuator):
    count = await actuator2.count(UserGroupPO.id > 0)
    if count == 0:
        for i in range(10):
            po = UserGroupPO()
            po.name = "userGroup_"+str(i)
            po.code = "000000000_"+str(i)
            po.add_assignment()
            actuator2.add_all(po)
    count = await actuator2.count(UserGroupPO.id > 0)
    return count
    pass


async def test_actuator_query_all(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO)
    mp = await actuator2.query_all_entity_mappings(stmt)
    print("userGroupPO", mp)
    assert len(mp) == count
    await actuator2.session.rollback()
    await actuator2.session.close_all()


async def test_actuator_query_all_entity(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO)
    mp: list[UserGroupPO] = await actuator2.query_all_entity(stmt)
    print("userGroupPO", mp)
    assert len(mp) == count
    for m in mp:
        assert isinstance(m, BasePO)
        print(m.name)
    await actuator2.session.rollback()
    await actuator2.session.close_all()


async def test_actuator_query_one_entity(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO)
    mp: UserGroupPO = await actuator2.query_one_entity(stmt)
    print("userGroupPO", mp)

    assert isinstance(mp, BasePO)
    await actuator2.session.rollback()
    await actuator2.session.close_all()


async def test_actuator_query_one_entity_mapping(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO)
    mp = await actuator2.query_one_entity_mapping(stmt)
    print("userGroupPO mapping", mp)

    assert isinstance(mp, dict)
    await actuator2.session.rollback()
    await actuator2.session.close_all()


async def test_actuator_select_manay(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO.id, UserGroupPO.name, UserGroupPO.code)
    mp = await actuator2.query_all_mappings(stmt)
    print("userGroupPO mapping", mp)
    assert len(mp) == count
    for m in mp:
        assert isinstance(m, dict)

    await actuator2.session.rollback()
    await actuator2.session.close_all()


async def test_actuator_select_one(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO.id)
    mp = await actuator2.query_one_mappings(stmt)
    print("userGroupPO mapping", mp)
    assert isinstance(mp, dict)
    await actuator2.session.rollback()
    await actuator2.session.close_all()
async def test_actuator_select_one_rows(db_service_param):
    cfg, Session, actuator = db_service_param
    actuator2: Actuator = actuator
    count = await add_user_group(actuator2)
    # all_entity_mappings
    stmt = Select(UserGroupPO.id)
    mp = await actuator2.query_all_mappings(stmt)
    assert len(mp) == count
    for m in mp:
        assert isinstance(m, dict)
        print("userGroupPO mapping", m)
    
    await actuator2.session.rollback()
    await actuator2.session.close_all()

 
async def test_dddd(db_service_param):
    '''
    userId=1
    userRolesSelect = (
        Select(UserRolePO.roleId).filter(
            UserRolePO.userId == UserPO.id, UserPO.id == userId)
    )
    userGroupRolesSelect = (
        Select(UserGroupRolePO.roleId).filter(
            UserGroupRolePO.userGroupId == UserPO.userGroupId, UserPO.id == userId)
    )
    sql= userRolesSelect.union(userGroupRolesSelect)
    ''' 
    pass
     