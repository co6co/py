
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import Request
from sanic.response import text, raw, redirect
from co6co_sanic_ext.utils import JSON_util
import json

from view_model.wx import wx_base_view
from model.filters.WxMenuFilterItems import WxMenuFilterItems
from model.pos.wx import WxMenuPO
from co6co_sanic_ext.model.res.result import Result
from sanic.response import redirect
from typing import List, Optional, Tuple
from co6co.utils import log
from datetime import datetime
from model.enum import wx_menu_state
from model.wx import pageAuthonParam
from view_model.aop.wx_auth_aop import WX_Oauth, WX_Oauth_debug
from view_model.wx import wx_authon_views
from services import getAccountName, getAppid
from wechatpy.pay import WeChatPay


class real_name_View(wx_authon_views):
    """
    实名认证接口
    """

    async def post(self, request):
        id = self.getUserId(request)
        accounName = await getAccountName(self.get_db_session(request), id)
        openId = await getAppid(self.get_db_session(request), accounName)
        client = self.cteate_wx_client(request, openId)
        pay = WeChatPay(appid, mch_id, api_key)
        user_info = pay.userinfo.query(openId)
        if user_info.get('errcode') is None:
            return self.response_json(user_info)
        else:
            return self.response_json(user_info)
