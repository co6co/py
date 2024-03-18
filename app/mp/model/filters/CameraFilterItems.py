from model.enum import TopicCategory, device_type
from model.pos.biz import bizCameraPO, bizMqttTopicPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbOperations
from co6co.utils import log
from sqlalchemy import or_, and_, Select
from sqlalchemy.orm import joinedload, contains_eager


class CameraFilterItems(absFilterItems):
    """
    AI盒子 过滤器
    """
    name: str = None
    code: str = None
    datetimes: list = None

    def __init__(self):
        super().__init__(bizCameraPO)
        self.listSelectFields = [
            bizCameraPO.id,
            bizCameraPO.uuid,
            bizCameraPO.code,
            bizCameraPO.innerIp,
            bizCameraPO.ip,
            bizCameraPO.name,
            bizCameraPO.no,
            bizCameraPO.siteId,
            bizCameraPO.cameraType,
            bizCameraPO.poster,
            bizCameraPO.streams,
            bizCameraPO.sip,
            bizCameraPO.channel1_sip,
            bizCameraPO.channel2_sip,
            bizCameraPO.channel3_sip,
            bizCameraPO.channel4_sip,
            bizCameraPO.channel5_sip,
            bizCameraPO.channel6_sip,
            bizCameraPO.channel7_sip,
            bizCameraPO.channel8_sip,
            bizCameraPO.channel9_sip,
            bizCameraPO.channel10_sip,
            bizCameraPO.createTime,
            bizCameraPO.updateTime,
            bizMqttTopicPO.topic.label("ptzTopic")
        ]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(bizCameraPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(bizCameraPO.code.__eq__(self.code))
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(bizCameraPO.createTime.between(
                self.datetimes[0], self.datetimes[1]))
        #ptz 主题
        filters_arr.append(or_(bizMqttTopicPO.category ==
                           TopicCategory.ptz.key, bizMqttTopicPO.category == None))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(*self.listSelectFields)
            .join_from(bizMqttTopicPO, onclause=bizMqttTopicPO.code == bizCameraPO.id)
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizCameraPO.createTime.desc(),)
