
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util
import json

from view_model.wx import wx_authon_views,wx_resposne
from model.filters.WxMenuFilterItems import  WxMenuFilterItems
from model.pos.wx import WxMenuPO
from co6co_sanic_ext.model.res.result import Result

from typing import List,Optional,Tuple
from co6co.utils import log  
from datetime import datetime
from model.enum import wx_menu_state

class template_message_View(wx_authon_views):
    """
    模板消息管理 似乎没太多用
    
    """
    async def get(self, request:Request): 
        client=self.cteate_wx_client(request,"wx181aa5d9ce286cf0")
        #demp= client.template.get("G6ockp-Y-qCZ1mBfvZrkmaFFboqMY4WvixLm0W90xZk")  
        #设置行业
        industry2:wx_resposne=client.template.set_industry(1,4) 
        #获取行业
        industry=client.template.get_industry()
        addResult=client.template.add(12345)
        #获取模板列表
        allTemplate=client.template.get_all_private_template()
        
        return JSON_util.response(Result.success({"industry":industry,"result":"addResult","tempList":allTemplate}))

