from model.enum import device_type
from model.pos.biz import bizRouterPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbOperations
from co6co.utils import log
from sqlalchemy import or_, and_, Select
from sqlalchemy.orm import joinedload, contains_eager


class RouterFilterItems(absFilterItems):
    """
    路由器 过滤器
    """
    name: str = None
    code: str = None
    datetimes: list = None

    def __init__(self):
        super().__init__(bizRouterPO)
        self.listSelectFields = [
            bizRouterPO.id,
            bizRouterPO.uuid,
            bizRouterPO.innerIp,
            bizRouterPO.ip,
            bizRouterPO.name,
            bizRouterPO.siteId,
            bizRouterPO.sim,
            bizRouterPO.ssd,
            bizRouterPO.password,
            bizRouterPO.createTime,
            bizRouterPO.updateTime,
        ]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(bizRouterPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(bizRouterPO.code.__eq__(self.code))
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(bizRouterPO.createTime.between(
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
        return (bizRouterPO.createTime.desc(),)
