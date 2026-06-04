from sanic.request import Request


from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from ..pos.other import sysDictTypePO


class Filter(absFilterItems):
    """
    字典类型 filter
    """
    name: str = None
    code: str = None

    def __init__(self):
        super().__init__(sysDictTypePO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []

        if self.checkFieldValue(self.name):
            filters_arr.append(sysDictTypePO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(sysDictTypePO.code.like(f"%{self.code}%"))

        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (sysDictTypePO.order.asc(),)
