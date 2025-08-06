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
from model.enum import DeviceCategory, DeviceCheckState


class DevicePO(UserTimeStampedModelPO):
    """设备表"""
    __tablename__ = "tb_device"
    id = Column("id", Integer, comment="主键",  autoincrement=True, primary_key=True)
    name = Column("name", String(64),  comment="设备名称")
    category = Column("category", Integer, comment=f"设备类型: {DeviceCategory.to_labels_str()}")
    code = Column("code", String(64),  comment="代码编码")
    serialNumber = Column("serial_number", String(128),  comment="设备序列号")
    vender = Column("vender", String(32),  comment="厂家")
    ip = Column("ip", String(64),  comment="设备IP", unique=True)
    lat = Column("lat", String(64),  comment="经度")
    lng = Column("lng", String(64),  comment="维度")
    userName = Column("user_name", String(64),  comment="对于个别设备，需要用户名和密码")
    passwd = Column("passwd", String(128),  comment="对于个别设备，需要用户名和密码")
    state = Column("state", Integer, comment="设备状态")
    checkState = Column("check_state", Integer, comment=f"检测状态{DeviceCheckState.to_labels_str()},如果有其他文字描述放置在描述check_desc")
    checkDesc = Column("check_desc",  String(64), comment="设备检测状态描述")
    checkTime = Column("check_time", DateTime,   comment="设备检测时间")
    checkImgPath = Column("check_img_path", String(255),   comment="检测后的视频文件")

    def update(self, po: DevicePO):
        self.ip = po.ip
        self.lat = po.lat
        self.lng = po.lng
        self.code = po.code
        self.name = po.name
        self.state = po.state
        self.vender = po.vender
        self.passwd = po.passwd
        self.category = po.category
        self.userName = po.userName
        self.serialNumber = po.serialNumber
