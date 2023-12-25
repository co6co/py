from model.pos.device import sysConfigPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple, List
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class ConfigFilterItems(absFilterItems):
    """
    设备类别过滤器
    """
    code: str
    name: str

    def __init__(self):
        super().__init__(sysConfigPO)
        self.listSelectFields = [sysConfigPO.id, sysConfigPO.code,
                                 sysConfigPO.name, sysConfigPO.value, sysConfigPO.createTime,sysConfigPO.updateTime]

    def filter(self) -> list:
        """
        过滤条件 
        """
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(sysConfigPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.code):
            filters_arr.append(sysConfigPO.code.like(f"%{self.code}%"))
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
        return (sysConfigPO.id.asc(),)
