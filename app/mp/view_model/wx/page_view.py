
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw,redirect
from co6co_sanic_ext.utils import JSON_util,json

from view_model.wx import wx_base_view
from model.pos.wx_where import WxMenuFilterItems,WxMenuPO
from co6co_sanic_ext.model.res.result import Result
from sanic.response import redirect
from typing import List,Optional,Tuple
from co6co.utils import log  
from datetime import datetime
from model.enum import wx_menu_state
from view_model.wx.fn_oauth import Authon_param,oauth

class Authon_View(wx_base_view): 
    @oauth
    @staticmethod
    def getAuthonInfo(request:Request,param,Authon_param):
        """
        为了 为了参数同一
        """
        return text("")
    
    def get(self,request:Request,appid:str):
        """
        微信服务器 redirect_uri  调用入口获取微信用户信息
        需要在公众号接口处配置回调地址：接口权限>网页服务>网页帐号>修改
            域名不能加http://
            测试公众号可以使用端口
        """  
        try:
            args=self.usable_args (request)
            code=args.get("code")
            state=args.get("state")  
            param=Authon_param(appid)
            param.__dict__.update(args)
            param.setState(state) 
            return Authon_View.getAuthonInfo(request,param)
        except Exception as e: 
            log.err(f"oauth2:异常:{e}")
            return redirect("/")

    def post(self,request:Request):
        """
        获取微信用户信息,数据库系统存储的授权信息
        """ 
        param=Authon_param()
        param.__dict__.update(request.json)
        code:str=param.code
        client=self.cteate_wx_client(request,param.appid) 

        