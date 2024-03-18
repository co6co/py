from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction

from sanic import Request
from model.filters.BoxFilterItems import BoxFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from view_model.base_view import AuthMethodView
from model.pos.biz import bizBoxPO
from co6co.utils import log
from sqlalchemy import Select
from typing import List, TypeVar
from view_model.biz.devices import getSiteId


class Boxs_View(AuthMethodView):
    """
    设备s:盒子
    """
    async def get(self, request: Request):
        """
        获取盒子列表 [{id:1,name:"boxName"}]
        """
        select = (
            Select(bizBoxPO.id, bizBoxPO.name)
        )
        return await self.query_mapping(request, select)

    async def post(self, request: Request):
        """
        列表 包括 相机列表，box信息
        """
        filterItems = BoxFilterItems()
        return await self.query_page(request, filterItems)
    async def patch(self, request: Request):
        siteId=getSiteId(request)
        filterItems = BoxFilterItems()
        select = (
            Select(*filterItems.listSelectFields).filter(bizBoxPO.siteId==siteId)
        )
        return await self.query_mapping(request, select,True)

    async def put(self, request: Request):
        """
        增加
        """
        po = bizBoxPO()
        return await self.add(request, po, self. getUserId(request))


class Box_View(AuthMethodView):
    """
    设备:盒子 
    """

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        po = bizBoxPO()

        async def settingValue(old: bizBoxPO, param: bizBoxPO):
            old.code = param.code
            old.innerIp = param.innerIp
            old.ip = param.ip
            old.name = param.name
            old.cpuNo = param.cpuNo
            old.mac = param.mac
            old.license = param.license
            old.talkbackNo= param.talkbackNo
        return await self.edit(request, pk, po, bizBoxPO, self. getUserId(request), settingValue)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, bizBoxPO)
