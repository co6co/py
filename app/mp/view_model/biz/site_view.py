from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction
import asyncio
from co6co_db_ext .db_operations import DbOperations
from co6co_db_ext.db_utils import db_tools
from sanic import Request, redirect
from sanic.response import text, raw, empty, file_stream
from co6co_sanic_ext.utils import JSON_util
import json
from model.enum import device_type, SiteConfigCategory
from model.filters.SiteFilterItems import SiteFilterItems, SiteDiveceFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from view_model import get_upload_path
from view_model.base_view import BaseMethodView, AuthMethodView
from model.pos.biz import bizCameraPO, bizRouterPO, bizBoxPO,  bizResourcePO, bizSitePo, bizSiteConfigPO
import os
import datetime
from co6co.utils import log
from sqlalchemy.engine.row import RowMapping
from sqlalchemy import Select
from typing import List
from sqlalchemy.sql import delete


async def editOrAddSiteConfig(request: Request, currentUserId: int, siteID: int, session: AsyncSession):
    configs: list = request.json.get("configs")
    removeConfig: List[int] = request.json.get("removeConfig")
    configPOList = []
    if configs != None:
        configPOList = [bizSiteConfigPO(**c) for c in configs]
    for cPo in configPOList:
        if cPo.id != None:
            oldPo: bizSiteConfigPO = await session.get_one(bizSiteConfigPO, cPo.id)
            oldPo.name = cPo.name
            oldPo.siteId = siteID
            oldPo.category = cPo.category
            oldPo.value = cPo.value
            oldPo.updateUser = currentUserId
            oldPo.updateTime = datetime.datetime.now()
        else:
            cPo.createUser = currentUserId
            cPo.siteId = siteID
            cPo.createTime = datetime.datetime.now()
            session.add(cPo)
    if removeConfig != None and len(removeConfig) > 0:
        stmt = delete(bizSiteConfigPO).where( bizSiteConfigPO.id.in_(removeConfig))
        await session.execute(stmt)


class Sites_View(AuthMethodView):
    """
    安全员站点s
    """
    async def get(self, request: Request):
        """
        获取站点列表 [{id:1,name:站点}]
        """ 
        select = (
            Select(bizSitePo.id, bizSitePo.name)
        )
        return await self.query_mapping(request, select) 
    
    async def patch(self, request: Request):
        """
        获取站点列表
        """
        filterItems = SiteFilterItems()
        filterItems.__dict__.update(request.json)
        try:
            async with self.get_db_session(request) as session, session.begin():
                session: AsyncSession = session
                total = await session.execute(filterItems.count_select)
                total = total.scalar()
                executer = await session.execute(filterItems.list_select)
                result = executer.mappings().all()
                result = [dict(a) for a in result]
                pageList = Page_Result.success(result, total=total)
                return JSON_util.response(pageList)
        except Exception as e:
            pageList = Page_Result.fail(message=f"请求失败：{e}")

    async def post(self, request: Request):
        """
        设备导航使用
        列表 包含 相机列表，box信息
        """
        filterItems = SiteDiveceFilterItems()
        filterItems.__dict__.update(request.json)
        # return JSON_util.response(Page_Result.fail())
        try:
            async with self.get_db_session(request) as session, session.begin():
                session: AsyncSession = session
                total = await session.execute(filterItems.count_select)
                total = total.scalar()
                executer = await session.execute(filterItems.list_select)
                result = executer.unique().scalars().all()
            data = []
            for a in result:
                d = {}
                a: bizSitePo = a
                d.update(a.to_dict2())
                devices = []
                if a.boxPO:
                    d.update({"box": a.boxPO.to_dict()})
                for pa in a.camerasPO:
                    pa: bizCameraPO = pa
                    dict = pa.to_dict()
                    if "streams" in dict and dict.get("streams") != None:
                        dict.update(
                            {"streams": json.loads(dict.get("streams"))})
                    devices.append(dict)
                d.update({"devices": devices})
                data.append(d)
            pageList = Page_Result.success(data, total=total)
        except Exception as e: 
            errorCode=log.generateCode(e)
            log.err(f"导航站点列表：{errorCode} ... ")
            pageList = Page_Result.fail(message=f"请求失败：{errorCode}")
        return JSON_util.response(pageList)

    async def put(self, request: Request):
        """
        增加站点
        """
        try:
            po = bizSitePo()
            po.__dict__.update(request.json)
            async with self.get_db_session(request) as session, session.begin():
                session.add(po)
                await session.flush()
                await editOrAddSiteConfig(request, self. getUserId(request), po.id, session)
            return JSON_util.response(Result.success())
        except Exception as e:
            log.err(e)
            return JSON_util.response(Result.fail(message=e))


class Site_View(AuthMethodView):
    """
    安全员站点
    """
    async def post(self, request: Request, pk: int):
        """
        获取详细信息内容
        """
        try:
            '''
            data=self.usable_args(request) 
            log.warn(data)
            data=data.get("category") 
            '''
            data = request.json.get("category")
            type: device_type = device_type.value_of(data)
            async with request.ctx.session as session, session.begin():
                session: AsyncSession = session
                if type == device_type.router:
                    select = (Select(bizRouterPO).where(
                        bizRouterPO.siteId == pk).order_by(bizRouterPO.id.asc()))
                if type == device_type.ip_camera:
                    select = (Select(bizCameraPO).where(
                        bizCameraPO.siteId == pk) .order_by(bizCameraPO.id.asc()))
                if type == device_type.box:
                    select = (Select(bizBoxPO).where(
                        bizBoxPO.siteId == pk).order_by(bizBoxPO.id.asc()))
                executer = await session.execute(select)
                poList = executer.scalars().all()
                result = db_tools.remove_db_instance_state(poList)
                return JSON_util.response(Result.success(result))
        except Exception as e:
            log.err(e)
            return JSON_util.response(Result.fail(message=e))

    async def put(self, request: Request, pk: int):
        """
        编辑站点
        """
        try:
            po = bizSitePo()
            po.__dict__.update(request.json)
            async with self.get_db_session(request) as session, session.begin():
                oldPo: bizSitePo = await session.get_one(bizSitePo, pk)
                if oldPo == None:
                    return JSON_util.response(Result.fail(message="未找到设备!"))
                oldPo.name = po.name
                oldPo.deviceCode = po.deviceCode
                oldPo.postionInfo = po.postionInfo
                oldPo.deviceDesc = po.deviceDesc
                oldPo.updateTime = datetime.datetime.now()
                # 编辑配置
                await editOrAddSiteConfig(request, self. getUserId(request), pk, session)

            return JSON_util.response(Result.success())
        except Exception as e:
            log.err("Err:", e)
            return JSON_util.response(Result.fail(message=e))


class Site_config_View(AuthMethodView):
    """
    安全员站点配置
    """
    async def post(self, request: Request, pk: int):
        """
        获取站配置
        """
        try:
            data = request.json.get("category")
            type: SiteConfigCategory = SiteConfigCategory.value_of(data)
            async with self.get_db_session(request) as session, session.begin():
                select = (
                    Select(bizSiteConfigPO.id, bizSiteConfigPO.name, bizSiteConfigPO.value,
                           bizSiteConfigPO.code, bizSiteConfigPO.category)
                    .where(bizSiteConfigPO.siteId == pk, bizSiteConfigPO.category == type.key).order_by(bizSiteConfigPO.id.asc())
                )
                executer = await session.execute(select)
                result = executer.fetchall()
                list = db_tools.list2Dict(result)
                return JSON_util.response(Result.success(list))
        except Exception as e:
            log.err(e)
            return JSON_util.response(Result.fail(message=e))
