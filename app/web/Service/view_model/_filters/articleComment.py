

from model.pos.business import articleCommentPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func, or_, and_, Select


class Filter(absFilterItems):
    """
    文章评论 filter
    """
    postId: int = None

    def __init__(self):
        super().__init__(articleCommentPO)

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.postId):
            filters_arr.append(
                articleCommentPO.postId.__eq__(self.postId))
        return filters_arr

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (articleCommentPO.createTime.asc(),)
