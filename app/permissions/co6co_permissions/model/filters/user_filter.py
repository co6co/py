
from ..pos.right import UserPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class user_filter(absFilterItems):
    """
        用户表过滤器
    """
    name: str = None
    userGroupId: int = None

    def __init__(self, userName=None, userGroupId: int = None):
        super().__init__(UserPO)
        self.name = userName
        self.userGroupId = userGroupId

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.userGroupId):
            filters_arr.append(UserPO.userGroupId.__eq__(self.userGroupId))
        if self.checkFieldValue(self.name):
            filters_arr.append(UserPO.userName.like(f"%{self.name}%"))
        return filters_arr

    def create_List_select(self):
        pass

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (UserPO.id.asc(),)
