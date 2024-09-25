

from model.pos.business import suggestPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    违法代码 filter
    """
    title: str = None
    category: int = None
    state: int = None

    def __init__(self):
        super().__init__(suggestPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.title):
            filters_arr.append(suggestPO.title.like(f"%{self.title}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(suggestPO.category.__eq__(self.category))
        if self.checkFieldValue(self.state):
            filters_arr.append(suggestPO.state.__eq__(self.state))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (suggestPO.id.desc(),)
