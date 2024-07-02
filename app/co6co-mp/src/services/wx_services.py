from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import Select
from co6co_db_ext .db_operations import DbOperations
from sanic import Request
from model import WechatConfig

from model.enum import Account_category, User_category, User_Group
from model.enum.wx import wx_event_type
from model.wx import wxUser
from model.pos.wx import WxUserPO
from co6co_permissions.model.pos.right import UserPO, AccountPO, UserGroupPO, RolePO
import datetime
from co6co.utils import log
from wechatpy import events
import uuid


async def createOrUpdateAccount(session: AsyncSession, wx_user: wxUser, accountStatus: str = None):
    """
    创建或更新微信用户相关表
    """
    async with session:
        opt = DbOperations(session)
        log.warn(f"userOpenId:{wx_user.openid}")
        select = (
            Select(WxUserPO)
            .options(joinedload(WxUserPO.accountPO))
            .filter(AccountPO.accountName == wx_user.openid, AccountPO.category == Account_category.wx.val)
        )
        po: WxUserPO = await opt._get_one(select, False)
        if po != None:
            if accountStatus != None:
                a: AccountPO = po.accountPO
                a.status = accountStatus
            wx_user.to(po)
        else:
            u_po = UserPO()
            u_po.category = User_category.unbind.val
            u_po.userName = wx_user.openid

            # 用戶組
            userGroupPO: UserGroupPO = await opt.get_one(UserGroupPO, UserGroupPO.code == User_Group.wx_user.name)
            if userGroupPO == None:
                raise Exception(f"数据库中未找到相关用户组{User_Group.wx_user.name}")
            u_po.userGroupPO = userGroupPO

            a_po = AccountPO()
            a_po.accountName = wx_user.openid
            a_po.category = Account_category.wx.val
            a_po.userPO = u_po
            a_po.uid = str(uuid.uuid4())

            w_po = WxUserPO()
            wx_user.to(w_po)
            w_po.accountPO = a_po
            opt.add(w_po)
        await opt.commit()


def wx_open_id_into_db(func):
    """
    关注/取消关注
    如果出现接收未关注事件，未能创建账号信息
    需要重新关注
    """
    async def warpper(request: Request, msg: events.BaseEvent, config: WechatConfig, *args, **kwargs):
        if msg.event == wx_event_type.subscribe.key or msg.event == wx_event_type.unsubscribe.key:
            wx_user = wxUser()
            wx_user.openid = msg.source
            wx_user.ownedAppid = config.appid
            try:
                await createOrUpdateAccount(request.ctx.session, wx_user, msg.event)
            except Exception as e:
                log.err(f"createOrUpdateAccount Error:{e}")
        return await func(request, msg, config, *args, **kwargs)
    return warpper


async def oauth_wx_user_to_db(request: Request, wx_user: wxUser):
    """
    微信用户入库
    """
    try:
        await createOrUpdateAccount(request.ctx.session, wx_user)
    except Exception as e:
        log.err(f"createOrUpdateAccount Error:{e}")
