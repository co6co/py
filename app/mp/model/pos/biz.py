
from co6co_db_ext.po import BasePO ,metadata,UserTimeStampedModelPO,TimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, Text, String,DateTime
from sqlalchemy.orm import Relationship 
from sqlalchemy.schema import DDL 
import uuid


class bizDevicePo(TimeStampedModelPO):
    """
    设备
    """
    __tablename__ = "biz_device" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用")
    deviceType= Column("device_type",Integer,comment="设备类型")
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    resourcesPO=Relationship("bizResourcePO",back_populates="devicePO",uselist=True,passive_deletes=True)
    boxPO=Relationship("bizBoxPO",uselist=False, back_populates="devicePo")
    cameraPO=Relationship("bizCameraPO",uselist=False, back_populates="devicePo")
    mqttPO=Relationship("bizMqttPO",uselist=False,  back_populates="devicePo")
    xttPO=Relationship("bizXssPO",uselist=False,  back_populates="devicePo")
    routerPO=Relationship("bizRouterPO",uselist=False,  back_populates="devicePo")

class bizBoxPO(UserTimeStampedModelPO):
    """
    盒子设备
    //UUID 根据实际需求需要更正
    //上传设备获得:RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60
    //配置BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60
    """
    __tablename__ = "biz_dev_box" 
    id = Column("id",ForeignKey(f"biz_device.id",ondelete="CASCADE"), primary_key=True)
    cpuNo = Column("cpu_serial_number",String(255)) 
    mac = Column("mac",String(128)) 
    license= Column("license",String(255)) 
    sip= Column("sip_address",String(64),comment="盒子SIP地址") 
    talkbackNo = Column("talkbackNo",Integer,comment="对讲号") 
   
    channel1_sip= Column("channel1_sip",String(64),comment="通道1 sip 地址") 
    channel2_sip= Column("channel2_sip",String(64),comment="通道2 sip 地址") 
    channel3_sip= Column("channel3_sip",String(64),comment="通道3 sip 地址") 
    devicePo=Relationship("bizDevicePo",back_populates="boxPO")  
class bizCameraPO(UserTimeStampedModelPO):
    """
    摄像头设备
    """
    __tablename__ = "biz_camera" 
    id = Column("id",ForeignKey(f"biz_device.id",ondelete="CASCADE"), primary_key=True)
    CameraType= Column("type",String(16))
    poster = Column("poster",String(255)) 
    streams = Column("stream_urls",String(2048),comment="json 对象[{url:xx,name:xx}]")  
    sip= Column("sip_address",String(64),comment="SIP地址") 
    talkbackNo = Column("talkbackNo",Integer,comment="对讲号")  
    channel1_sip= Column("channel1_sip",String(64),comment="通道1 sip 地址") 
    channel2_sip= Column("channel2_sip",String(64),comment="通道2 sip 地址") 
    channel3_sip= Column("channel3_sip",String(64),comment="通道3 sip 地址") 
    devicePo=Relationship("bizDevicePo",back_populates="cameraPO")  
    def __repr__(self) -> str:
        return f"{self.__class__} id:{self.id},streams:{self.streams},createTime:{self.createTime}"
 

class bizMqttPO(UserTimeStampedModelPO):
    """
    Mqtt 服务器
    """
    __tablename__ = "biz_dev_mqttPO" 
    id = Column("id",ForeignKey(f"biz_device.id",ondelete="CASCADE"), primary_key=True)
    tcpPort = Column("tcpPort",Integer) 
    wsPort = Column("wsPort",Integer)
    wssPort = Column("wssPort",Integer) 
    devicePo=Relationship("bizDevicePo",back_populates="mqttPO") 
 

class bizXssPO(UserTimeStampedModelPO):
    """
    Xss 服务器
    """
    __tablename__ = "biz_dev_xssPO" 
    id = Column("id",ForeignKey(f"biz_device.id",ondelete="CASCADE"), primary_key=True)
    port = Column("port",Integer) 
    sip = Column("sip",String(20))
    domain= Column("domain",String(10))
    password= Column("password",String(32)) 
    devicePo=Relationship("bizDevicePo",back_populates="xttPO") 

class bizRouterPO(UserTimeStampedModelPO):
    """
    4G路由器
    """
    __tablename__ = "biz_dev_router" 
    id = Column("id",ForeignKey(f"biz_device.id",ondelete="CASCADE"), primary_key=True)
    sim = Column("sim",String(20)) 
    ssd= Column("wifi_ssd",String(32))
    password= Column("wifi_password",String(32))   
    devicePo=Relationship("bizDevicePo",back_populates="routerPO") 
    
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
    result= Column("result",Text,comment="告警Result结果")
    media= Column("media",String(2048),comment="告警media结果")
    gps= Column("gps",String(2048),comment="告警gps结果")

    alarmPO=Relationship(bizAlarmPO,back_populates="alarmAttachPO")
    
