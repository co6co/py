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


class resourcePO(TimeStampedModelPO):
    """
    资源
    """
    __tablename__ = "sys_resource"
    id = Column("id", BigInteger, comment="主键", autoincrement=True, primary_key=True)
    uid = Column("uuid", String(36),  unique=True, default=uuid.uuid1())
    category = Column("category", Integer, comment="资源类型:0:图片资源,1:视频资源, 2:文件")
    subCategory = Column("sub_category", Integer, comment="子资源类型")
    hash = Column("hash", String(256), unique=True, comment="使用hash256计算值")
    url = Column("url_path", String(255), comment="资源路径,针对根路径下的绝对路径")
    metaName = Column("meta_name", String(255), comment="名称")
    size = Column("size", BigInteger, comment="大小")
    # , passive_deletes=True ->当删除资源是 所有的用户资源已会被删除
    userResourceList = Relationship("userResourcePO", back_populates="userResource")


class userResourcePO(TimeStampedModelPO):
    """
    用户资源资源
    """
    __tablename__ = "user_resource"
    id = Column("id", BigInteger, comment="主键", autoincrement=True, primary_key=True)
    resourceId = Column("resource_id", ForeignKey("{}.{}".format(resourcePO.__tablename__, resourcePO.id.name)))
    ownUserId = Column("own_user_id", BigInteger, comment="所属用户")
    name = Column("name", String(255), comment="名称")
    userResource = Relationship("resourcePO", back_populates="userResourceList")
