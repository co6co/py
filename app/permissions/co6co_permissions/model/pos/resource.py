from __future__ import annotations
from co6co_db_ext.po import BasePO, TimeStampedModelPO, UserTimeStampedModelPO, CreateUserStampedModelPO
from sqlalchemy import func, INTEGER, Integer, UUID,  INTEGER, BigInteger, Column, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship, declarative_base, Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL
from sqlalchemy import MetaData
import uuid


class bizResourcePO(TimeStampedModelPO):
    """
    资源
    """
    __tablename__ = "sys_resource"
    id = Column("id", BigInteger, comment="主键",
                autoincrement=True, primary_key=True)
    uid = Column("uuid", String(36),  unique=True, default=uuid.uuid1())
    category = Column("category", Integer, comment="资源类型:0:图片资源,1:视频资源")
    subCategory = Column("sub_category", Integer, comment="子资源类型")
    url = Column("url_path", String(255), comment="资源路径,针对根路径下的绝对路径")
