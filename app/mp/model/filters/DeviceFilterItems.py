from model.enum import device_type
from model.pos.biz import  bizCameraPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbOperations
from co6co.utils import log
from sqlalchemy import or_, and_, Select
from sqlalchemy.orm import joinedload, contains_eager
import json

 
class DeviceFilterItems(absFilterItems):
    """
    设备管理
    """
    name: str = None
    category: int = None
    datetimes: list = None

    def __init__(self):
        super().__init__(bizCameraPO)
        self.listSelectFields = [
            bizCameraPO.id, bizCameraPO.uuid, bizCameraPO.name, bizCameraPO.cameraType, bizCameraPO.createTime, bizCameraPO.ip, bizCameraPO.innerIp,
            bizCameraPO.streams,bizCameraPO.poster
        ]

    def filter(self) -> list:
        """
        过滤条件 
        """ 
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(bizCameraPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(bizCameraPO.cameraType.__eq__(self.category))
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(bizCameraPO.createTime.between(
                self.datetimes[0], self.datetimes[1]))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(*self.listSelectFields)
            .outerjoin(bizCameraPO)
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizCameraPO.createTime.desc(),)


class CameraFilterItems(absFilterItems):
    """
    ip相机
    """
    name: str

    def __init__(self):
        super().__init__(bizCameraPO) 

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = [] 
        if self.checkFieldValue(self.name):
            filters_arr.append(bizCameraPO.name.like(f"%{self.name}%"))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(bizCameraPO) 
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizCameraPO.id.asc(),)


class posterTaskFilterItems(absFilterItems):
    """
    定时任务过滤条件
    """

    def __init__(self, userName=None, role_id: int = None):
        super().__init__(bizCameraPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        filters_arr.append(bizCameraPO.streams.is_not(None))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(bizCameraPO)  # .join(device.deviceCategoryPO,isouter=True)
            .filter(and_(*self.filter()))
        )
        return select
        return

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizCameraPO.id.asc(),)
