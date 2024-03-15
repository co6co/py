from model.pos.biz import bizSitePo,  bizBoxPO,  bizAlarmPO, bizAlarmTypePO, bizAlarmAttachPO
from sqlalchemy .orm.attributes import InstrumentedAttribute
from typing import Tuple, List
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import or_, and_, Select
from sqlalchemy.orm import defaultload, joinedload, contains_eager


class AlarmCategoryFilterItems(absFilterItems):
    """
    告警类型
    """

    def __init__(self):
        super().__init__(bizAlarmTypePO)
        self.listSelectFields = [bizAlarmTypePO.alarmType, bizAlarmTypePO.desc]

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        return filters_arr

    def create_List_select(self):
        select = (Select(*self.listSelectFields))
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizAlarmPO.id.asc(),)


class AlarmFilterItems(absFilterItems):
    """
    告警
    """
    alarmType: str = None
    siteId: int = None
    datetimes: List[str]

    def __init__(self):
        self.datetimes = []
        super().__init__(bizAlarmPO)

        self.listSelectFields =[bizAlarmPO.id,bizSitePo.id.label("siteId"),
                                bizAlarmPO.uuid.label("uid"),
                                 bizAlarmPO.alarmType,
                                 bizAlarmPO.videoUid,
                                 bizAlarmPO.markedImageUid,  
                                 bizAlarmPO.alarmTime, 
                                 bizAlarmPO.rawImageUid,  
                                 bizAlarmPO.createTime,
                                 bizBoxPO.name.label("boxName"),
                                 bizSitePo.name.label("siteName"),
                                 bizAlarmTypePO.desc.label("alarmTypeDesc")
                                ] 
    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.alarmType):
            filters_arr.append(bizAlarmPO.alarmType.__eq__(self.alarmType))
        if self.checkFieldValue(self.siteId):
            filters_arr.append(bizSitePo.id==self.siteId)
        if self.datetimes and len(self.datetimes) == 2:
            filters_arr.append(bizAlarmPO.alarmTime.between(
                self.datetimes[0], self.datetimes[1]))
        return filters_arr

    def create_List_select(self): 
        select = ( 
            Select(*self.listSelectFields)
            .join(bizAlarmTypePO, isouter=True, onclause=bizAlarmPO.alarmType == bizAlarmTypePO.alarmType)
            .join(bizBoxPO, isouter=True, onclause=bizAlarmPO.boxId == bizBoxPO.id)
            .join(bizSitePo, isouter=True, onclause=bizSitePo.id == bizBoxPO.siteId) 
            .filter(*self.filter())
        ) 
        return select

    def getDefaultOrderBy(self) -> Tuple[InstrumentedAttribute]:
        """
        默认排序
        """
        return (bizAlarmPO.alarmTime.desc(),)
