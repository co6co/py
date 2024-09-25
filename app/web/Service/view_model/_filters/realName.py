

from model.pos.business import realNameAuthenPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    实名认证信息 filter
    """
    pid: int = None
    name: str = None
    code: str = None

    def __init__(self):
        super().__init__(realNameAuthenPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(realNameAuthenPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(realNameAuthenPO.identityNumber.like(f"%{self.code}%"))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (realNameAuthenPO.order.asc(),)
