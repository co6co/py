from model.pos import device
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple, List
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class  CategoryFilterItems(absFilterItems):
    """
    设备类别过滤器
    """

    def __init__(self):
        super().__init__(device.deviceCategoryPO)
        self.listSelectFields = [device.deviceCategoryPO.id,
                                 device.deviceCategoryPO.name, device.deviceCategoryPO.code]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        return filters_arr

    def create_List_select(self):
        select = (
            Select(*self.listSelectFields)
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (device.deviceCategoryPO.id.asc(),)


class DeviceFilterItems(absFilterItems):
    """
    设备表过滤器
    """
    name: str = None
    category: int = None
    datetimes: list = None

    def __init__(self):
        super().__init__(device.devicePo)
        self.listSelectFields = [
            device.devicePo.id, device.devicePo.ip, device.devicePo.name, device.devicePo.code,
            device.devicePo.createTime,
            device.deviceCategoryPO.name.label(
                "categoryName"), device.deviceCategoryPO.code.label("categoryCode")
        ]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(device.devicePo.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(device.devicePo.category.__eq__(self.category))
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(device.devicePo.createTime.between(
                self.datetimes[0], self.datetimes[1]))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(*self.listSelectFields).join(device.deviceCategoryPO, isouter=True)
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (device.devicePo.id.asc(),)


class LightFilterItems(DeviceFilterItems):
    """
    补光灯设置
    """
    allows:    bool = True
    startTime: str = None
    endTime:   str = None 
    def __init__(self):
        super().__init__()
        self.listSelectFields=[device.devicePo]
