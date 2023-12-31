from sanic import Sanic, Blueprint, Request
from sanic.response import json, file_stream, file

import datetime
from sanic.response import json
from sanic import Blueprint, Request
from sanic import exceptions
from model.pos.biz import bizXssPO
from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction

from co6co_web_db.utils import DbJSONEncoder as JSON_util
import json as sys_json
from model.filters.UserFilterItems import UserFilterItems
from co6co_db_ext.res.result import Result, Page_Result

from services import authorized, generateUserToken
from co6co.utils import log
from co6co_db_ext .db_operations import DbOperations, DbPagedOperations, and_, joinedload
from sqlalchemy import func, text
from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools

server_api = Blueprint("server_API")


@server_api.route("/server/getxssConfig", methods=["POST",])
@authorized
async def xss(request: Request):
    """
    获取对讲服务器
    """ 
    async with request.ctx.session  as session,session.begin():
        session: AsyncSession = session 
        select = (
            Select(bizXssPO.port, bizXssPO.ip, bizXssPO.name) 
        )
        result = (await session.execute(select)).fetchone()   
    return JSON_util.response(Result.success(data=db_tools.row2dict(result), message="获取成功"))
