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

from view_model.topic import Topic_View

top_api = Blueprint("topic_API", url_prefix="/topic")  
top_api.add_route(Topic_View.as_view(),"/<category:str>/<code:str>",name=Topic_View.__name__)