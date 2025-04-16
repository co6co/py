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

 



class DevicePO(UserTimeStampedModelPO):
    __tablename__ = "tb_device"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="代码名称")
    code = Column("code", String(64),  comment="代码编码")
    ip = Column("ip", String(64),  comment="代码编码")
    lat = Column("lat", String(64),  comment="经度")
    lng = Column("lng", String(64),  comment="维度")
    userName = Column("user_name", String(64),  comment="对于个别设备，需要用户名和密码")
    passwd = Column("passwd", String(128),  comment="对于个别设备，需要用户名和密码")
    state = Column("state", Integer, comment="设备状态")

    def update(self, po: DevicePO):
        self.name = po.name
        self.code = po.code
        self.ip = po.ip
        self.lat = po.lat
        self.lng = po.lng
        self.userName = po.userName
        self.passwd = po.passwd
        self.state = po.state
