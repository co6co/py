from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction
 
from sanic import Request 
from model.filters.RouterFilterItems import RouterFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from view_model.base_view import AuthMethodView
from model.pos.biz import bizRouterPO 
from co6co.utils import log
from sqlalchemy import Select
from typing import List
from view_model.biz.devices import getSiteId


class Routers_View(AuthMethodView):
    """
    设备s: 路由器
    """
    async def get(self, request: Request):
        """
        获取列表 [{id:1,name:"boxName"}]
        """
        select = (
            Select(bizRouterPO.id, bizRouterPO.name)
        )
        return await self.query_mapping(request, select)

    async def post(self, request: Request):
        """
        列表 
        """
        filterItems = RouterFilterItems()
        return await self.query_page(request, filterItems)
    async def patch(self, request: Request):
        """
        通过 siteId 获取 路由dev
        """
        siteId=getSiteId(request)
        filterItems = RouterFilterItems()
        select = (
            Select(*filterItems.listSelectFields).filter(bizRouterPO.siteId==siteId)
        )
        return await self.query_mapping(request, select,True)

    async def put(self, request: Request):
        """
        增加
        """
        po = bizRouterPO()
        return await self.add(request, po, self. getUserId(request))


class Router_View(AuthMethodView):
    """
    设备:路由器 
    """

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        po = bizRouterPO() 
        def settingValue(old: bizRouterPO, param: bizRouterPO): 
            old.innerIp = param.innerIp
            old.ip = param.ip 
            old.innerIp=param.innerIp
            old.ip=param.ip
            old.name=param.name
            old.sim =param.sim
            old.ssd=param.ssd
            old.password=param.password
        
        return await self.edit(request, pk, po, bizRouterPO, self. getUserId(request), settingValue)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, bizRouterPO)
