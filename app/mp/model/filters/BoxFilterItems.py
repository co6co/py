from model.enum import device_type
from model.pos.biz import bizBoxPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbOperations
from co6co.utils import log
from sqlalchemy import or_, and_, Select
from sqlalchemy.orm import joinedload, contains_eager


class BoxFilterItems(absFilterItems):
    """
    AI盒子 过滤器
    """
    name: str = None
    code: str = None
    datetimes: list = None

    def __init__(self):
        super().__init__(bizBoxPO)
        self.listSelectFields = [
            bizBoxPO.id,
            bizBoxPO.siteId,
            bizBoxPO.uuid,
            bizBoxPO.code,
            bizBoxPO.innerIp,
            bizBoxPO.ip,
            bizBoxPO.name,
            bizBoxPO.cpuNo,
            bizBoxPO.mac,
            bizBoxPO.license,
            bizBoxPO.talkbackNo,
            bizBoxPO.createTime,
            bizBoxPO.updateTime,
        ]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(bizBoxPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(bizBoxPO.code.__eq__(self.code))
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(bizBoxPO.createTime.between(
                self.datetimes[0], self.datetimes[1]))
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
        return (bizBoxPO.id.asc(),)
