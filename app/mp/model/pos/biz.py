
from co6co_db_ext.po import BasePO ,metadata,UserTimeStampedModelPO,TimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import Relationship 
from sqlalchemy.schema import DDL 
import uuid


class bizDevicePo(TimeStampedModelPO):
    """
    盒子设备
    """
    __tablename__ = "biz_device" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用")
    deviceType= Column("device_type",Integer,comment="设备类型")
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    resourcesPO=Relationship("bizResourcePO",back_populates="devicePO",uselist=True,passive_deletes=True)

class bizResourcePO(BasePO):
    """
    资源
    """
    __tablename__ = "biz_resource"
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    uid = Column("uuid",String(36),  unique=True,default=uuid.uuid1())
    category = Column("category",Integer,comment="资源类型:0:图片资源,1:视频资源") 
    subCategory = Column("sub_category",Integer,comment="子资源类型")
    url = Column("url_path",String(255),comment="资源路径,针对根路径下的绝对路径")
    createTime=Column("create_time",DateTime , server_default=func.now())  
    deviceId = Column("device_id",ForeignKey(f"{bizDevicePo.__tablename__}.{bizDevicePo.id.name}",ondelete="CASCADE"),nullable=False,index=True)
    devicePO=Relationship("bizDevicePo",back_populates="resourcesPO")

class bizAlarmTypePO(BasePO): 
    __tablename__ = "biz_alarm_type"
    alarmType = Column("alarm_type", String(64),comment= "告警类型", primary_key=True)
    desc= Column("desc",String(128), comment= "告警描述")
    createTime=Column("create_time",DateTime , server_default=func.now() ) 
    updateTime= Column("update_time", DateTime,comment="修改时间")  
    alarmPOs=Relationship("bizAlarmPO",back_populates="alarmTypePO",uselist=True,passive_deletes=True)


class bizAlarmPO(BasePO):
    __tablename__ = "biz_alarm"
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    uuid = Column("uuid",String(64),unique=True,comment= "全局ID盒子上传")
    alarmType= Column("alarm_type",ForeignKey(f"{bizAlarmTypePO.__tablename__}.{bizAlarmTypePO.alarmType.name}",ondelete="CASCADE"))
   
    videoUid= Column("video_resource_uid",String(36),comment= "视频资源")
    rawImageUid= Column("image_raw_resource_uid",String(36),comment= "原始图片资源")
    markedImageUid= Column("image_marked_resource_uid",String(36),comment= "标注图片资源")
    attachResource1= Column("attach_resource1_uid",String(36),comment= "附加资源1")
    attachResource2= Column("attach_resource2_uid",String(36),comment= "附加资源2")
    attachResource3= Column("attach_resource3_uid",String(36),comment= "附加资源3")
    
    taskSession= Column("task_session_id",String(64),comment= "任务属性")
    taskDesc= Column("task_desc",String(64),comment= "任务描述") 
    alarmTime=Column("alarm_time",DateTime ,comment= "告警事件" )  
    createTime=Column("create_time",DateTime , server_default=func.now() ) 

    alarmTypePO=Relationship(bizAlarmTypePO,back_populates="alarmPOs") 
    alarmAttachPO=Relationship("bizAlarmAttachPO",back_populates="alarmPO",uselist=False,passive_deletes=True)

class bizAlarmAttachPO(BasePO):
    __tablename__ = "biz_alarm_attach" 
    id= Column("alarm_id",ForeignKey(f"{bizAlarmPO.__tablename__}.{bizAlarmPO.id.name}",ondelete="CASCADE"),comment="主键id",primary_key=True)
    result= Column("result",String(2048),comment="告警Result结果")
    media= Column("media",String(2048),comment="告警media结果")
    gps= Column("gps",String(2048),comment="告警gps结果")

    alarmPO=Relationship(bizAlarmPO,back_populates="alarmAttachPO")
    
