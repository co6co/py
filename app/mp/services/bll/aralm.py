
from services.bll import baseBll
from model.pos.biz import bizAlarmTypePO 
from model.pos.right import UserPO,WxUserPO,AccountPO,UserGroupPO

from sqlalchemy import Select
from co6co.utils import log
from co6co_db_ext.db_utils import db_tools
from model.enum import User_Group

class alarm_bll(baseBll): 
    async def get_subscribe_alarm_user(self,ownedAppid:str):
        """
        获取需订阅告警用户
        """
        try:  
            async with self.session as session,session.begin():  
                select=( 
                    Select(UserPO.id,UserPO.userName ,WxUserPO.openid,WxUserPO.nickName)
                    .join_from(WxUserPO,AccountPO,isouter=True,onclause=AccountPO.uid==WxUserPO.accountUid)  
                    
                    .join(UserPO,isouter=True,onclause=AccountPO.userId==UserPO.id) 
                    .join(UserGroupPO,isouter=True,onclause=UserPO.userGroupId==UserGroupPO.id)  
                    .filter(WxUserPO.ownedAppid== ownedAppid,UserGroupPO.code==User_Group.wx_alarm.key) 
                )   
                executer=await session.execute(select)  
                result = executer.mappings().fetchall() 
                result = [dict(a) for a in result]  
            return result 
        except Exception as e: 
            log.err(f"获取需订阅告警用户失败：{e}") 
            return None 
        
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
     
    