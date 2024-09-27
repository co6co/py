from __future__ import annotations
from co6co_db_ext.po import BasePO, UserTimeStampedModelPO, TimeStampedModelPO
from sqlalchemy import func, INTEGER, DATE, FLOAT, DOUBLE, SMALLINT, Integer, UUID, Text, INTEGER, BigInteger, Column, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, declarative_base, Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL
from sqlalchemy import MetaData
from sqlalchemy.dialects.mysql import VARCHAR


class TaskPO(UserTimeStampedModelPO):
    """
    任务，定时任务，
    后台自动执行
    """
    __tablename__ = "tb_task"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="任务名称")
    code = Column("code", String(64),  comment="任务编码")
    category = Column("category", Integer, comment="0:系统,10:用户")
    # 增加触发器涉及的逻辑较多，使用 cron 表达式 可以完全替代相关需求
    # trigger = Column("trigger", String(16), comment="date|interval|cron")
    cron = Column("cron", String(128), comment="cron表达式")
    state = Column("state", Integer, comment="任务状态")
    sourceCode = Column("source_code", String(4096), comment="执行代码")
    execStatus = Column("status", Integer, comment="执行状态")

    def update(self, po: TaskPO):
        self.name = po.name
        self.category = po.category
        self.cron = po.cron
        self.state = po.state
        self.sourceCode = po.sourceCode
        self.execStatus = po.execStatus
