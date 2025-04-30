

from model.pos.tables import DevicePO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    违法代码 filter
    """
    name: str = None
    code: str = None
    state: int = None
    ip: str = None
    category: int = None

    def __init__(self):
        super().__init__(DevicePO)
        self.name = None
        self.code = None
        self.state = None
        self.ip = None
        self.category = None

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(DevicePO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.ip):
            filters_arr.append(DevicePO.ip.like(f"%{self.ip}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(DevicePO.code.like(f"%{self.code}%"))
        if self.checkFieldValue(self.state):
            filters_arr.append(DevicePO.state.__eq__(self.state))
        if self.checkFieldValue(self.category):
            filters_arr.append(DevicePO.category.__eq__(self.category))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (DevicePO.createTime.desc(),)
