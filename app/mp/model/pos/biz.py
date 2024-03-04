
from co6co_db_ext.po import BasePO ,metadata,UserTimeStampedModelPO,TimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, Text, String,DateTime
from sqlalchemy.orm import Relationship 
from sqlalchemy.schema import DDL 
import uuid 


class bizSitePo(TimeStampedModelPO):
    """
    站点信息
    一个站点由一个box 和若干个相机组成，一个 4G 路由器，组成 
    """
    __tablename__ = "biz_site" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    name= Column("name",String(64),nullable=False, comment="站点名称")
    deviceCode= Column("device_code",String(64),nullable=False, comment="设备编号")
    postionInfo= Column("postion_info",String(255),nullable=False, comment="设备安装位置")
    deviceDesc= Column("device_desc",String(255),nullable=False, comment="设备使用描述")
   
    boxPO =Relationship("bizBoxPO",uselist=False, back_populates="sitePO") 
    routerPO=Relationship("bizRouterPO",uselist=False,back_populates="sitePO") 
    camerasPO=Relationship("bizCameraPO",uselist=True, back_populates="sitePO") 

class bizBoxPO(UserTimeStampedModelPO):
    """
    盒子设备
    //UUID 根据实际需求需要更正
    //上传设备获得:RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60
    //配置BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60
    """
    __tablename__ = "biz_dev_box" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    siteId=Column("site_id",ForeignKey(f"biz_site.id" ))

    uuid = Column("uuid",String(64),comment="设备唯一标识，主要与设备通行使用")  
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 

    cpuNo = Column("cpu_serial_number",String(255)) 
    mac = Column("mac",String(128)) 
    license= Column("license",String(255)) 
    sip= Column("sip_address",String(64),comment="盒子SIP地址") 
    talkbackNo = Column("talkbackNo",Integer,comment="对讲号") 
   
    channel1_sip= Column("channel1_sip",String(64),comment="通道1 sip 地址") 
    channel2_sip= Column("channel2_sip",String(64),comment="通道2 sip 地址") 
    channel3_sip= Column("channel3_sip",String(64),comment="通道3 sip 地址") 

    innerConfigUrl= Column("inner_config_url",String(2048),comment="设备配置URL") 
    configUrl= Column("config_url",String(2048),comment="设备配置URL")  
    sshConfigUrl= Column("ssh_config_url",String(2048),comment="SSH配置")  
    
    resourcesPO=Relationship("bizResourcePO", uselist=True,  back_populates="boxPO")
    sitePO =Relationship("bizSitePo",  back_populates="boxPO")
    alarmPO=Relationship("bizAlarmPO",back_populates="boxPO" )
    
class bizCameraPO(UserTimeStampedModelPO):
    """
    摄像头设备
    """
    __tablename__ = "biz_camera" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用") 
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    no = Column("no",Integer,comment="球机编号")
    siteId=Column("site_id",ForeignKey(f"biz_site.id" ) )
    cameraType= Column("type",String(16))
    poster = Column("poster",String(255)) 
    streams = Column("stream_urls",String(2048),comment="json 对象[{url:xx,name:xx}]")  
    sip= Column("sip_address",String(64),comment="SIP地址") 
    talkbackNo = Column("talkbackNo",Integer,comment="对讲号")  
    channel1_sip= Column("channel1_sip",String(64),comment="通道1 sip 地址") 
    channel2_sip= Column("channel2_sip",String(64),comment="通道2 sip 地址") 
    channel3_sip= Column("channel3_sip",String(64),comment="通道3 sip 地址") 
    channel4_sip= Column("channel4_sip",String(64),comment="通道4 sip 地址") 
    channel5_sip= Column("channel5_sip",String(64),comment="通道5 sip 地址") 
    channel6_sip= Column("channel6_sip",String(64),comment="通道6 sip 地址") 
    channel7_sip= Column("channel7_sip",String(64),comment="通道7 sip 地址") 
    channel8_sip= Column("channel8_sip",String(64),comment="通道8 sip 地址") 
    channel9_sip= Column("channel9_sip",String(64),comment="通道9 sip 地址") 
    channel10_sip= Column("channel10_sip",String(64),comment="通道10 sip 地址") 

    innerConfigUrl= Column("inner_config_url",String(2048),comment="设备配置URL") 
    configUrl= Column("config_url",String(2048),comment="设备配置URL")  
       
    sitePO=Relationship("bizSitePo",back_populates="camerasPO")
    
    def __repr__(self) -> str:
        return f"{self.__class__} id:{self.id},streams:{self.streams},createTime:{self.createTime}"
 

class bizMqttPO(UserTimeStampedModelPO):
    """
    Mqtt 服务器
    """
    __tablename__ = "biz_svr_mqtt" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用") 
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
 
    tcpPort = Column("tcpPort",Integer) 
    wsPort = Column("wsPort",Integer)
    wssPort = Column("wssPort",Integer)  

class bizSipPO(UserTimeStampedModelPO):
    """
    sip 服务器
    """
    __tablename__ = "biz_svr_sip" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用") 
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    port = Column("port",Integer) 
    sip = Column("sip",String(20))
    domain= Column("domain",String(10))
    password= Column("password",String(32))  
 

class bizXssPO(UserTimeStampedModelPO):
    """
    Xss 服务器
    """
    __tablename__ = "biz_svr_xss" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用") 
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    port = Column("port",Integer)  
    userName= Column("user_name",String(10))
    password= Column("password",String(32))  

class bizRouterPO(UserTimeStampedModelPO):
    """
    4G路由器
    """
    __tablename__ = "biz_dev_router" 
    id = Column("id",Integer, primary_key=True,autoincrement=True)
    uuid = Column("device_uuid",String(64),comment="设备唯一标识，主要与设备通行使用") 
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="设备名称") 
    siteId=Column("site_id",ForeignKey(f"biz_site.id" ))

    sim = Column("sim",String(20)) 
    ssd= Column("wifi_ssd",String(32))
    password= Column("wifi_password",String(32))    

    innerConfigUrl= Column("inner_config_url",String(2048),comment="设备配置URL") 
    configUrl= Column("config_url",String(2048),comment="设备配置URL")  

    sitePO=Relationship("bizSitePo",back_populates="routerPO") 
    
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
    boxId = Column("box_id",ForeignKey("biz_dev_box.id",ondelete="CASCADE"),nullable=False,index=True) 
    boxPO=Relationship("bizBoxPO",back_populates="resourcesPO")

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
    boxId = Column("device_id",ForeignKey(f"{bizBoxPO.__tablename__}.{bizBoxPO.id.name}"),comment="产生记录的设备" )
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
    boxPO=Relationship("bizBoxPO",back_populates="alarmPO",uselist=False )

class bizAlarmAttachPO(BasePO):
    __tablename__ = "biz_alarm_attach" 
    id= Column("alarm_id",ForeignKey(f"{bizAlarmPO.__tablename__}.{bizAlarmPO.id.name}",ondelete="CASCADE"),comment="主键id",primary_key=True)
    result= Column("result",Text,comment="告警Result结果")
    media= Column("media",String(2048),comment="告警media结果")
    gps= Column("gps",String(2048),comment="告警gps结果")

    alarmPO=Relationship(bizAlarmPO,back_populates="alarmAttachPO")
    
