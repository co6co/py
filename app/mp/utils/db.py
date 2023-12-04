from functools import wraps
from co6co.utils import log
from wechatpy import WeChatClient
from sanic.request import Request
from sanic.response import redirect,raw
from model import WechatConfig 
from wechatpy import messages ,events
from co6co_db_ext.db_operations import DbOperations
from model.pos.right import   AccountPO
from model.enum import Account_category
from model.enum.wx import wx_event_type
import asyncio,datetime

def wx_open_id_into_db(func):
    """
    如果出现接收未关注事件，未能创建账号信息
    需要重新关注
    """
    async def warpper(request:Request,msg:events.BaseEvent,config:WechatConfig,*args, **kwargs): 
        if msg.event== wx_event_type.subscribe.getName()  or msg.event== wx_event_type.unsubscribe.getName(): 
            async with request.ctx.session as session:  
                operation=DbOperations(session) 
                isExist=await operation.exist(AccountPO.accountName==msg.source,AccountPO.category==Account_category.wx.getValue(), AccountPO.attachInfo==config.appid,column=AccountPO.accountName)
                if isExist: 
                    po:AccountPO=await operation.get_one(AccountPO,AccountPO.accountName==msg.source,AccountPO.category==Account_category.wx.getValue(), AccountPO.attachInfo==config.appid)
                    po.status=msg.event 
                    po.updateTime=datetime.datetime.now()
                else:
                    po=AccountPO() 
                    po.accountName=msg.source
                    po.category=Account_category.wx.getValue()
                    po.attachInfo=config.appid
                    po.status=msg.event
                    operation.add(po)
                await operation.commit() 
        return await func(request,msg,config,*args, **kwargs)
    return warpper