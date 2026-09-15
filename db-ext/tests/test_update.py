from co6co_db_ext.db_utils import DbCallable
from co6co_db_ext.actuator import Actuator
from co6co_db_ext.session import session_context
from co6co_db_ext.db_utils import Select,Update
from sqlalchemy import text
from tests.right import RecordPO 
async def test_one_data(db_service_param):
    _, factory, actuator = db_service_param 
    actuator:Actuator = actuator

    po = RecordPO()
    po.id = 1
    po.name = "test"
    po.state = "success"
    po.previousStatus = None
    po.ipAddress = "192.168.1.1"
    po.webBrowser = "chrome"
    po.message = "test message"
    actuator.add_all(po) 
    await actuator.session.flush()
    #sql = Update(RecordPO).where(RecordPO.id == 1).values({RecordPO.name:"test_1",RecordPO.state:"failed"})
    #sql = Update(RecordPO).where(RecordPO.id == 1).ordered_values((RecordPO.name,"test_1"),(RecordPO.state,"failed"))
    sql = Update(RecordPO).where(RecordPO.id == 1).values({RecordPO.previousStatus:RecordPO.state,RecordPO.state:"failed"})
    await actuator.execSQL(sql)
    await actuator.session.flush()
    print(po.previousStatus)

    sql = Update(RecordPO).where(RecordPO.id == 1).ordered_values((RecordPO.previousStatus,RecordPO.state),(RecordPO.state,"failed"))
    await actuator.execSQL(sql)
    await actuator.session.flush()
    print(po.previousStatus)
    await actuator.session.rollback()



    


