from co6co_db_ext.actuator import Actuator
import pytest
from co6co_db_ext.db_session import db_service, connectSetting
from .right import *

 
async def test_actuator(db_service_param):
    cfg, Session, actuator =  db_service_param
    print("test_actuator",cfg)
    #po = UserPO()
    #po.userName = "admin" 
    #actuator.add_all([po])
    #assert actuator.all() == [po]
    pass