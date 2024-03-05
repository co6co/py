# -*- encoding:utf-8 -*-
import datetime
import json
from sanic import Blueprint,Request
from sanic import exceptions
from model.pos.right import UserPO ,AccountPO ,UserGroupPO,RolePO,UserRolePO
from model.pos.wx import WxUserPO
from sqlalchemy.ext.asyncio import AsyncSession


from co6co_sanic_ext.utils import JSON_util
from model.filters.UserFilterItems import UserFilterItems 
from co6co_db_ext.res.result import Result,Page_Result 

from services import authorized,generateUserToken
from co6co.utils import log
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sqlalchemy import func,text
from sqlalchemy.sql import Select
from sanic.request.parameters import RequestParameters
import requests
from cacheout import Cache

xss_api = Blueprint("xss_API", url_prefix="/xss")  


async def listForCache(request:Request):
    cache:Cache=request.app.ctx.Cache 
    sipList=[]
    ks=cache.keys()
    for k in ks: 
        v=cache.get(k) 
        if v!=None:sipList.append({"uri":k})   
    return JSON_util.response(Result.success(sipList))

@xss_api.route("/list",methods=["GET",])
async def list(request:Request):
    dev=request.get_args( ).get("dev")
    atv=request.get_args( ).get("atv")
    url=None
    if dev=="gb": 
        url="https://stream.jshwx.com.cn:8441/list?dev=gb" 
        return await listForCache(request)
    if atv!=None: url=f"https://stream.jshwx.com.cn:8441/list?atv={atv}" 
    if url==None:return JSON_util.response(Result.fail(message="未知操作"))
    res=requests.get(url)
    return JSON_util.response(json.loads(res.text)) 
