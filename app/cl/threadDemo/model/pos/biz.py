from __future__ import annotations
from co6co_db_ext.po import BasePO, UserTimeStampedModelPO, TimeStampedModelPO
from sqlalchemy import func, INTEGER, DATE, FLOAT, DOUBLE, SMALLINT, Integer, UUID, Text, INTEGER, BigInteger, Column, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, declarative_base, Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
import datetime
from sqlalchemy.schema import DDL
from sqlalchemy import MetaData
from sqlalchemy.dialects.mysql import VARCHAR
from model.enum import FlowCategory


class DeviceFlowPO(UserTimeStampedModelPO):
    """设备客流表"""
    __tablename__ = "tb_device_flow"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    ip = Column("ip", String(64),  comment="ip 地址")
    category = Column("category", Integer, comment=f"客流数据类型: {FlowCategory.to_labels_str()}")
    enterNum = Column("enter_num", Integer, comment="进入数")
    leaveNum = Column("leave_num", Integer, comment="离开数")
    passingNum = Column("passing_num", Integer, comment="经过数")
    duplicatePeople = Column("duplicate_people_num", Integer, comment="重复数")
    childEnterNum = Column("child_enter_num", Integer, comment="小孩进入数")
    childLeaveNum = Column("child_leave_num", Integer, comment="小孩离开数")
    startTime = Column("start_time", DateTime,   comment=f"统计开始时间,category={FlowCategory.statFlow.val}时有效")
    endTime = Column("end_time", DateTime,   comment=f"统计结束时间,category={FlowCategory.statFlow.val}时有效")

    def update(self, po: DeviceFlowPO):
        self.id = po.id
        self.ip = po.ip
        self.category = po.category
        self.enterNum = po.enterNum
        self.leaveNum = po.leaveNum
        self.passingNum = po.passingNum
        self.duplicatePeople = po.duplicatePeople
        self.childEnterNum = po.childEnterNum
        self.childLeaveNum = po.childLeaveNum
        self.startTime = po.startTime
        self.endTime = po.endTime

    def to_dev_data(self):

        return {
            "enterNum": self.enterNum,
            "leaveNum": self.leaveNum,
            "passingNum": self.passingNum,
            "duplicatePeople": self.duplicatePeople,
            "childEnterNum": self.childEnterNum,
            "childLeaveNum": self.childLeaveNum
        }


class DeviceFlowTotalityPO(TimeStampedModelPO):
    """设备客流表"""
    __tablename__ = "tb_device_flow_totality"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    totalNum = Column("max_total", Integer, comment="在园数当前小数最大的在园数")
    currentHour = Column("current_hour", DateTime,   comment=f"小时数", unique=True)

    def update(self, po):
        self.totalNum = po.totalNum
        self.edit_assignment()

    def setCurrentHour(self):
        now = datetime.datetime.now()
        truncated_time = now.replace(minute=0, second=0, microsecond=0)
        self.currentHour = truncated_time
