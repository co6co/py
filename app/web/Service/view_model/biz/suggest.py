
from sanic.response import text
from sanic import Request

from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from view_model.wx import wx_authon_views
from model.pos.business import suggestPO, suggestResourcePO, suggestReplyPO
from view_model._filters.suggest import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute
import json
from co6co_sanic_ext.utils import JSON_util
import datetime
from co6co.utils import log
from services.cache import wxCacheData
from model.enum import suggest_state


class Views(AuthMethodView):
    async def get(self, request: Request):
        return JSON_util.response(Result.success(data=suggest_state.to_dict_list()))

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        param.__dict__.update(request.json)

        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = suggestPO()
        userId = self.getUserId(request)
        # todo resourceIds

        async def beforeFun(po: suggestPO, session, request):
            if type(po.otherInfo) == dict or type(po.otherInfo) == list:
                po.otherInfo = json.dumps(po.otherInfo)

        return await self.add(request, po,   userId=userId, beforeFun=beforeFun)


class View(wx_authon_views):
    routePath = "/<pk:int>"

    async def get(self, request: Request, pk: int):
        """
       获取详情
        """
        select = (
            Select(
                suggestPO.id,
                suggestPO.category,
                suggestPO.title,
                suggestPO.otherInfo,
                suggestPO.contents,
                suggestPO.state
            ).filter(suggestPO.id == pk)
        )
        return await self.get_one(request, select,  isPO=False)

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        async def beforeFun(oldPo: suggestPO, po: suggestPO, session, request):
            if type(po.otherInfo) == dict or type(po.otherInfo) == list:
                oldPo.otherInfo = json.dumps(po.otherInfo)
        return await self.edit(request, pk, suggestPO,  userId=self.getUserId(request), fun=beforeFun)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        args = self.usable_args(request)
        msg: str = None

        async def sendTemplageInfo(oldPo: suggestPO, session: AsyncSession):
            log.warn(args)
            reason = args.get("reason", "未输入删除原因")
            nonlocal msg
            msg = self.sendTemplateMessage(request, oldPo.createUser, "删除{}".format(oldPo.title), reason)

        async def afterFun(oldPo, session, request):
            log.warn("msaage", msg)
            if msg != None:
                return Result.success(message=msg)

        return await self.remove(request, pk, suggestPO, beforeFun=sendTemplageInfo, afterFun=afterFun)


class Reply_Views(wx_authon_views):
    """
    增加回复
    """

    routePath = "/reply/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        增加回复
        pk :建议ID
        对象有{state,title,content}
        """
        userid = self.getUserId(request)
        select = Select(suggestPO).options(selectinload(suggestPO.replyList)).filter(suggestPO.id == pk)

        async def beforeFun(oldPo: suggestPO, _: suggestPO, session, request: Request):
            po = suggestReplyPO()
            po.__dict__.update(request.json)
            po.createUser = userid
            po.createTime = datetime.datetime.now()
            status = request.json.get("state", suggest_state.finished.val)
            # desc = request.json.get("stateDesc", "状态")
            oldPo.state = status
            oldPo.replyList.append(po)
            msg = await self.sendTemplateMessage(request, oldPo.createUser,  po.title, po.content)
        result = await self.edit(request, select, suggestPO,  userId=userid, fun=beforeFun, json2Po=False)

        return result
