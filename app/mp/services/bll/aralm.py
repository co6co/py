
from services.bll import baseBll
from model.pos.biz import bizAlarmTypePO 
from sqlalchemy import Select
from co6co.utils import log
from co6co_db_ext.db_utils import db_tools
class alarm_bll(baseBll): 
    async def get_alram_type_desc(self,alarmType:str):
        """
        获取告警类型描述
        """
        try:  
            async with self.session as session,session.begin():   
                select=(
                    Select(bizAlarmTypePO).where(bizAlarmTypePO.alarmType==alarmType)
                )  
                executer=await session.execute(select)  
                result:bizAlarmTypePO = executer.fetchone()
                if result==None:log.warn(f"未能获取告警类型‘{alarmType}’的描述")  
                else :return db_tools.row2dict(result) 
                return None
        except Exception as e: 
            log.err(f"获取告警类型描述出错：{e}") 
            return None 
     
    