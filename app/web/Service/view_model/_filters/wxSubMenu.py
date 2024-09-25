

from model.pos.wx import WxSubMenuPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    微信子菜单 filter
    """
    name: str = None
    category: int = None
    state: int = None

    def __init__(self):
        super().__init__(WxSubMenuPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(WxSubMenuPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(WxSubMenuPO.category.__eq__(self.category))
        if self.checkFieldValue(self.state):
            filters_arr.append(WxSubMenuPO.state.__eq__(self.state))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (WxSubMenuPO.order.asc(),)
