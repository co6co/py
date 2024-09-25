

from model.pos.business import articlePO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    文章信息 filter
    """
    title: str = None
    category: int = None

    def __init__(self):
        super().__init__(articlePO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.title):
            filters_arr.append(articlePO.title.like(f"%{self.title}%"))
        if self.checkFieldValue(self.category):
            filters_arr.append(
                articlePO.category.__eq__(self.category))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (articlePO.order.asc(),)
