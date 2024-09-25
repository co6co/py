

from model.pos.business import locationInforPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    位置信息 filter
    """
    pid: int = None
    name: str = None
    category: int = None

    def __init__(self):
        super().__init__(locationInforPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(locationInforPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(
                locationInforPO.category.__eq__(self.category))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (locationInforPO.order.asc(),)
