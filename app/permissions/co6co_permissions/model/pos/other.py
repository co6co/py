from __future__ import annotations
from co6co_db_ext.po import BasePO, TimeStampedModelPO, UserTimeStampedModelPO, CreateUserStampedModelPO
from sqlalchemy import func, INTEGER, SmallInteger, Integer, UUID,  INTEGER, BigInteger, Column, ForeignKey, String, DateTime, CheckConstraint
from sqlalchemy.orm import relationship, declarative_base, Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL
from sqlalchemy import MetaData
import uuid


class sysConfigPO(UserTimeStampedModelPO):
    """
    系统配置
    """
    __tablename__ = "sys_config"
    id = Column("id", Integer, autoincrement=True, primary_key=True)
    name = Column("name", String(64))
    code = Column("code", String(64),  unique=True)
    dictFlag = Column("dict_flag", String(1),  comment="Y:使用字典做配置,N:手动配置")
    dictTypeId = Column("dict_type_id", Integer,  comment="字典类型ID")
    value = Column("value", String(1024),  comment="配置值")
    remark = Column("remark", String(2048), comment="备注")


class sysDictTypePO(UserTimeStampedModelPO):
    """
    字典类型
    """
    __tablename__ = "sys_dict_type"
    id = Column("id", Integer, autoincrement=True, primary_key=True)
    name = Column("name", String(64))
    code = Column("code", String(64),  unique=True)
    desc = Column("desc", String(1024))
    sysFlag = Column("sys_flag", String(1), comment="系统标识:Y/N")
    # py 层限制 取值范围为:(0-1)
    state = Column("state", SmallInteger,  CheckConstraint(
        'state >= 0 AND state <= 1'), comment="状态:0/1->禁用/启用",)
    order = Column("order", Integer, comment="排序")


class sysDictPO(UserTimeStampedModelPO):
    """
    字典
    """
    __tablename__ = "sys_dict"
    id = Column("id", Integer, autoincrement=True, primary_key=True)
    dictTypeId = Column("dict_type_id", Integer, comment="字典类型ID")
    name = Column("name", String(64))
    code = Column("code", String(64))
    desc = Column("desc", String(1024))
    # py 层限制 取值范围为:(0-1)
    state = Column("state", SmallInteger,  CheckConstraint(
        'state >= 0 AND state <= 1'), comment="状态:0/1->禁用/启用",)
    order = Column("order", Integer, comment="排序")


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
