
from services.bll import baseBll
from model.pos.right import UserPO, UserGroupPO, AccountPO
from model.pos.wx import WxUserPO
from sqlalchemy import Select
from co6co.utils import log
from model.enum import User_Group
import asyncio


class wx_user_bll(baseBll):
    async def get_subscribe_alarm_user(self, ownedAppid: str):
        """
        获取需订阅告警用户
        """
        try:
            async with self.session as session, session.begin():
                select = (
                    Select(UserPO.id, UserPO.userName,
                           WxUserPO.openid, WxUserPO.nickName)
                    .join_from(WxUserPO, AccountPO, isouter=True, onclause=AccountPO.uid == WxUserPO.accountUid)

                    .join(UserPO, isouter=True, onclause=AccountPO.userId == UserPO.id)
                    .join(UserGroupPO, isouter=True, onclause=UserPO.userGroupId == UserGroupPO.id)
                    .filter(WxUserPO.ownedAppid == ownedAppid, UserGroupPO.code == User_Group.wx_alarm.key)
                )
                executer = await session.execute(select)
                result = executer.mappings().fetchall()
                result = [dict(a) for a in result]
            return result
        except Exception as e:
            log.err(f"获取需订阅告警用户失败：{e}")
            return None
