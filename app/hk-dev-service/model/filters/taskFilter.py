from typing import Tuple
from model.pos.device import TasksPO
from model.pos.right import UserPO
from co6co_db_ext.db_filter import absFilterItems, Select, and_
from sqlalchemy .orm import joinedload
import co6co_db_ext.db_operations
from sqlalchemy .orm.attributes import InstrumentedAttribute


class TaskFilterItems(absFilterItems):
    """
    任务 表过滤器
    """
    name: str = None
    status: int = None
    user: int = None

    def __init__(self, name=None, status: int = None, user: int = None):
        super().__init__(TasksPO)
        self.listSelectFields = [TasksPO.id, TasksPO.name,UserPO.userName,
                                 TasksPO.data, TasksPO.createTime]
        self.name = name
        self.status = status
        self.user = user

    def filter(self) -> list:
        filters_arr = []
        if self.checkFieldValue(self.name):
            filters_arr.append(TasksPO.name.like(f"%{self.name}%"))
        if self.checkFieldValue(self.status):
            filters_arr.append(TasksPO.status.__eq__(self.status))
        if self.checkFieldValue(self.user):
            filters_arr.append(TasksPO.createUser.__eq__(self.user))
        return filters_arr

    def create_List_select(self):
        select = (
            Select(*self.listSelectFields).outerjoin(UserPO, UserPO.id == TasksPO.createUser)
            .filter(and_(*self.filter()))
        )
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        return (TasksPO.id.desc(),)
