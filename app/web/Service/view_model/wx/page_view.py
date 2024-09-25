
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


class Authon_View(wx_base_view):
    """
    页面认证 
    """

    def get_wx_authon_param(self, request: Request, appid: str) -> pageAuthonParam:
        """
        从参数中获取认证参数
        """
        args = self.usable_args(request)
        log.warn(args)
        code = args.get("code")
        state = args.get("state")

        param = pageAuthonParam(appid)
        param.__dict__.update(args)
        return param

    @WX_Oauth
    @staticmethod
    def getAuthonInfo(request: Request, param: pageAuthonParam):
        """
        为了 为了参数同一
        """
        return text("")

    def get(self, request: Request, appid: str):
        """
        微信服务器 redirect_uri  调用入口获取微信用户信息
        需要在公众号接口处配置回调地址：
            接口权限>网页服务>网页帐号>修改
                域名不能加http://
                测试公众号可以使用端口号
        """
        try:
            param: pageAuthonParam = self.get_wx_authon_param(request, appid)
            return Authon_View.getAuthonInfo(request, param)
        except Exception as e:
            log.err(f"oauth2:异常:{e}")
            return text(e, 500)

    def post(self, request: Request):
        """
        获取微信用户信息,数据库系统存储的授权信息
        """
        param = pageAuthonParam()
        param.__dict__.update(request.json)
        code: str = param.code
        # client = self.cteate_wx_client(request, param.appid)


class Authon_debug_View(Authon_View):
    """
    调试使用
    """
    @WX_Oauth_debug
    @staticmethod
    def getAuthonInfo(request: Request, param, Authon_param):
        """
        为了 为了参数同一
        """
        return text("")

    def get(self, request: Request, appid: str):
        """
        微信服务器 redirect_uri  调用入口获取微信用户信息
        需要在公众号接口处配置回调地址：接口权限>网页服务>网页帐号>修改
            域名不能加http://
            测试公众号可以使用端口
        """
        try:
            log.warn("页面授权DEBUG页...")
            param: pageAuthonParam = self.get_wx_authon_param(request, appid)
            return Authon_debug_View.getAuthonInfo(request, param)
        except Exception as e:
            log.err(f"oauth2:异常:{e}")
            return redirect("/")
