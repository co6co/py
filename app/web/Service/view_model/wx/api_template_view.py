
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import Request
from sanic.response import text, raw
from co6co_sanic_ext.utils import JSON_util
from co6co_db_ext.db_utils import DbCallable, db_tools
from sqlalchemy.sql import Select, Delete

from view_model.wx import wx_authon_views
from co6co_sanic_ext.model.res.result import Result

from typing import List, Optional, Tuple
from co6co.utils import log
from model.pos.wx import WxTemploatePO
from model.filters.wxTemplateFilter import Filter


class template_sync_View(wx_authon_views):
    """
    同步模板数据
    """
    routePath = "/sync/<appId:str>"

    async def put(self, request: Request, appId: str):
        """
        同步模板信息
        从微信服务器获取模板 保存到数据库
        """
        client = self.cteate_wx_client(request, appId)
        if client == None:
            return self.response_json(Result.fail(message="未找'{}',对应的配置信息".format(appId)))

        call = DbCallable(self.get_db_session(request))

        async def exec(session: AsyncSession):
            flag = 0
            templateIdArray = []
            addArray = []
            removeRount = 0
            for item in allTemplate.get("template_list", []):
                flag += 1
                newPo = WxTemploatePO.convert(item)
                newPo.ownedAppid = appId
                templateIdArray.append(newPo.templateId)
                oldPo: WxTemploatePO = await db_tools.execForPo(session, Select(WxTemploatePO).filter(WxTemploatePO.ownedAppid == appId, WxTemploatePO.templateId == newPo.templateId))
                if oldPo != None:
                    oldPo.update(item)
                else:
                    addArray.append(newPo)
            if len(addArray) > 0:
                session.add_all(addArray)
            if len(templateIdArray) > 0:
                deleteSql = Delete(WxTemploatePO).filter(WxTemploatePO.ownedAppid == appId, WxTemploatePO.templateId.not_in(templateIdArray))
                removeRount = await db_tools.execSQL(session, deleteSql)

            return JSON_util.response(Result.success(data={"count": flag, "remove": removeRount}, message="同步成功!"))
        try:
            # demp= client.template.get("G6ockp-Y-qCZ1mBfvZrkmaFFboqMY4WvixLm0W90xZk")
            # 设置行业
            # industry2:wx_resposne=client.template.set_industry(1,4)
            # 获取行业
            # industry=client.template.get_industry()
            # addResult=client.template.add(12345)
            # 获取模板列表
            allTemplate = client.template.get_all_private_template()
            return await call(exec)
        except Exception as e:
            log.err("同步失败", e)
            return JSON_util.response(Result.fail(str(e),  message=f"同步失败!"))


class template_Views(wx_authon_views):
    """
    模板消息管理  
    """

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()
        param.__dict__.update(request.json)

        return await self.query_page(request, param, isPO=False)


class template_View(wx_authon_views):
    routePath = "/app/<appId:str>"

    async def get(self, request: Request, appId: str):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(WxTemploatePO.templateId, WxTemploatePO.title).filter(WxTemploatePO.ownedAppid == appId)
            .order_by(WxTemploatePO.createTime.desc())
        )
        return await self.query_list(request, select,  isPO=False)


class template_pk_View(wx_authon_views):
    routePath = "/<pk:int>"

    async def get(self, request: Request, pk: int):
        """
        获取详细信息
        """
        select = (
            Select(WxTemploatePO).filter(WxTemploatePO.id == pk)
        )
        return await self.get_one(request, select,  isPO=True, remove_db_instance=True)
