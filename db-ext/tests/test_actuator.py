from co6co_db_ext.actuator import Actuator
import pytest  
from co6co_db_ext.db_session import db_service,connectSetting
from .right import *
def test_actuator(db_service_param):
    cfg,Session,actuator:Actuator = db_service_param
    po= UserPO()
    po.userName="admin"
    actuator



   
