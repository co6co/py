

from sanic import Request

from sqlalchemy.ext.asyncio import AsyncSession

from co6co_sanic_ext.utils import JSON_util, sys_json
from co6co.utils import log
from model.filters.DeviceFilterItems import DeviceFilterItems, CameraFilterItems
from co6co_db_ext.db_operations import DbOperations,joinedload
from view_model.base_view import AuthMethodView
from model.pos.biz import bizDevicePo, bizCameraPO
from model.enum import device_type
from model.params.devices import cameraParam, streamUrl
from co6co_sanic_ext.model.res.result import Result, Page_Result
from sqlalchemy.sql import Select


import os
from utils import createUuid
import datetime
from sanic.response import file, empty, json
from sqlalchemy.engine.row import RowMapping

from view_model.biz.poster_view import Image_View


class Devices_View(AuthMethodView):
    """
    设备API
    """
    async def get(self, request: Request):
        """
        获取设备类型
        """
        dictList = device_type.to_dict_list()
        return JSON_util.response(Result.success({"deviceType": dictList}))

    async def post(self, request: Request):
        """
        获取相机设备 list 
        """
        param = DeviceFilterItems()
        param.__dict__.update(request.json)
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer = await session.execute(param.count_select)
            total = executer.scalar()
            result = await session.execute(param.list_select)
            result = result.mappings().all()
            result = [dict(a) for a in result]
            pageList = Page_Result.success(result, total=total)
            await session.commit()
        return JSON_util.response(pageList)


class IP_Cameras_View(AuthMethodView):

    async def put(self, request: Request):
        """
        增加相机
        """
        try:
            param = cameraParam()
            param.__dict__.update(request.json)
            id=self.getUserId(request)
            async with request.ctx.session as session:
                session: AsyncSession = session
                po = bizCameraPO()
                po.streams = sys_json.dumps(param.streamUrls)
                po.createTime=datetime.datetime.now()
                po.createUser=id
                devicePo = bizDevicePo()
                devicePo.uuid=createUuid()
                devicePo.deviceType = param.deviceType
                devicePo.name = param.name
                devicePo.innerIp = param.innerIp
                devicePo.cameraPO = po
                devicePo.createTime=datetime.datetime.now()
                session.add(po)
                await session.commit()
            return JSON_util.response(Result.success())
        except Exception as e:
            return JSON_util.response(Result.fail(message=e))

    async def post(self, request: Request):
        """
        获取相机设备 list 
        """
        param = CameraFilterItems()
        param.__dict__.update(request.json)
        
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer = await session.execute(param.count_select)
            total = executer.scalar()
            result = await session.execute(param.list_select)
            result = result.mappings().all()
            for a in result:
                # sqlalchemy.engine.row.RowMapping
                log.succ(type(a.get("streams")))
                log.succ(a.get("streams"))
                log.succ(a.get("name"))
            result = [dict(a) for a in result]
            pageList = Page_Result.success(result, total=total)
            await session.commit()
        return JSON_util.response(pageList)


class IP_Camera_poster_View(Image_View):
    async def get(self, request: Request, pk: int):
        """
        相机 poster 
        """
        async with request.ctx.session as session:
            session: AsyncSession = session
            one: bizCameraPO = await session.get_one(bizCameraPO, pk)
            if one != None and one.poster != None and os.path.exists(one.poster):
                return await file(one.poster, mime_type="image/jpeg")
            await session.commit()
        return empty(status=404)


class IP_Camera_View(AuthMethodView):

    async def put(self, request: Request, pk: int):
        """
        编辑相机
        """
        try:
            param = cameraParam()
            param.__dict__.update(request.json)
            id=self.getUserId(request)
            async with request.ctx.session as session:
                session: AsyncSession = session
                select = (
                    Select(bizCameraPO) 
                    .options(joinedload(bizCameraPO.devicePo)).filter(bizCameraPO.id==pk)
                )
                execute=await session.execute(select) 
                po:bizCameraPO=execute.scalar()
                if po == None: return JSON_util.response(Result.fail(message="未找到设备!"))
                log.warn(po)
                po.streams = sys_json.dumps(param.streamUrls)
                po.updateTime=datetime.datetime.now()
                po.updateUser=id
                devicePo: bizDevicePo = po.devicePo
                devicePo.deviceType = param.deviceType
                devicePo.name = param.name
                devicePo.innerIp = param.innerIp
                devicePo.updateTime=datetime.datetime.now() 
                await session.commit()
            return JSON_util.response(Result.success())
        except Exception as e: 
            return JSON_util.response(Result.fail(message=e))
