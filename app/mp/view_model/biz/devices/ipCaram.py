from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction

from sanic import Request
from model.filters.CameraFilterItems import CameraFilterItems
from co6co_sanic_ext.model.res.result import Result, Page_Result

from model.params.devices import cameraParam
from view_model.base_view import AuthMethodView
from model.pos.biz import bizCameraPO, bizMqttTopicPO
from co6co.utils import log
from sqlalchemy import Select, or_
from typing import List
from view_model.biz.devices import getSiteId
from utils import createUuid
from model.enum import TopicCategory
from datetime import datetime


async def updatePtzTopic(po: bizCameraPO, session: AsyncSession, request: Request):
    log.start_mark("update topic ...")
    param = cameraParam()
    param.__dict__.update(request.json)
    
    select = (
        Select(bizMqttTopicPO).filter(bizMqttTopicPO.code == po.id,
                                      bizMqttTopicPO.category == TopicCategory.ptz.key)
    )
    exec = await session.execute(select)
    topicPO: TopicCategory = exec.scalar()
    id = request.ctx.current_user["id"]
    if topicPO == None:
        topicPO: bizMqttTopicPO = bizMqttTopicPO()
        topicPO.code = po.id
        topicPO.mqttServerId=1 # //todo
        topicPO.createTime = datetime.now()
        topicPO.createUser = id
        topicPO.category = TopicCategory.ptz.key
        topicPO.topic = param.ptzTopic
        session.add(topicPO)
    else:
        topicPO.topic = param.ptzTopic
        topicPO.updateTime = datetime.now()
        topicPO.updateUser = id
    log.end_mark("lllll")


class IpCameras_View(AuthMethodView):
    """
    设备s: 监控球机
    """
    async def get(self, request: Request):
        """
        获取列表 [{id:1,name:"deviceName"}]
        """
        print(request.args, type(request.args))
        siteId = self.usable_args(request).get("siteId") 
        select = (
            Select(bizCameraPO.id, bizCameraPO.name).filter(
                bizCameraPO.siteId == siteId)
        )
        return await self.query_mapping(request, select)

    async def post(self, request: Request):
        """
        列表 
        """
        filterItems = CameraFilterItems()
        return await self.query_page(request, filterItems)

    async def patch(self, request: Request):
        """
        获取站点下所有的监控球机
        """
        siteId = getSiteId(request)
        filterItems = CameraFilterItems() 
        select = (
            Select(*filterItems.listSelectFields)
            .join(bizMqttTopicPO, onclause=bizMqttTopicPO.code == bizCameraPO.id, isouter=True)
            .filter(bizCameraPO.siteId == siteId, or_(bizMqttTopicPO.category == TopicCategory.ptz.key, bizMqttTopicPO.category == None))
        )
        return await self.query_mapping(request, select)

    async def put(self, request: Request):
        """
        增加
        """
        po = bizCameraPO()
        param = cameraParam()
        param.__dict__.update(request.json)

        async def settingValue(po: bizCameraPO,session,request):
            log.succ("before00")
            param.set_po(po)
            po.uuid = createUuid()

        return await self.add(request, po, self. getUserId(request), settingValue, updatePtzTopic)
        # ptz 主题


class IpCamera_View(AuthMethodView):
    """
    设备:监控球机 
    """

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        try:
            po = bizCameraPO()
            param = cameraParam()
            param.__dict__.update(request.json)

            async def settingValue(old: bizCameraPO, po: bizCameraPO, session, request: Request):
                print("23333")
                param.set_po(old)
                if old.uuid == None:
                    old.uuid = createUuid()
                print("1111111111111111111")
                await updatePtzTopic(old, session, request)

            return await self.edit(request, pk, po, bizCameraPO, self. getUserId(request), settingValue)
        except Exception as e:
            raise e

    async def patch(self, request: Request, pk: int):
        """
        获取 
        """
        filterItems = CameraFilterItems()
        select = (
            Select(*filterItems.listSelectFields)
            .join(bizMqttTopicPO, onclause=bizMqttTopicPO.code == bizCameraPO.id, isouter=True)
            .filter(bizCameraPO.id == pk)
        )
        return await self.query_mapping(request, select, oneRecord=True)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, bizCameraPO)
