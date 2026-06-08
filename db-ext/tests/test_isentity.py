from co6co_db_ext.actuator import Actuator

from .right import UserPO
from sqlalchemy import Select

def test_isentity(): 
    stmt=Select(UserPO).where(UserPO.id==1)
    assert Actuator.is_entity_select(stmt)
    stmt=Select(UserPO.id).where(UserPO.id==1)
    assert not Actuator.is_entity_select(stmt)
    stmt=Select(UserPO.id,UserPO.userName).where(UserPO.id==1)
    assert not Actuator.is_entity_select(stmt)