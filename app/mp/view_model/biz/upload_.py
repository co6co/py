
from decimal import Decimal
from model.pos.biz import bizAlarmPO,bizAlarmAttachPO
from datetime import datetime
import json

class Response:
    """
    基础响应
    """
    Code:int
    Desc:str

    @classmethod
    def success(cls, message:str="上传成功") : 
        obj :Response= object.__new__(cls)  
        obj.Code=0
        obj.Desc=message
        return obj
    @classmethod
    def fail(cls, message:str="上传失败"):
        obj:Response=object.__new__(cls) 
        obj.Code=500
        obj.Desc=message
        return obj

class Video_Response(Response):
    VideoId:str

class Box_base_Param:
    BoardId:str
    BoardIp:str   
    GBDeviceId:str
    GBTaskChnId:str
    ip:str


class Video_Param(Box_base_Param):
    """
    盒子上传视频参数
    """  
    Video:str
    
class Alert_Param(Box_base_Param):
    """
    盒子上传视频参数
    """
    UniqueId:str
    Summary:str # 告警类型 
    VideoFile:str
    Result:dict
    Media:dict
    GPS:dict  
    TimeStamp:int 
    Addition:str
    AlarmId:str
    #原始图片
    ImageData:str 
    #标注过图片
    ImageDataLabeled:str
   
    LocalRawPath:str
    LocalLabeledPath:str 
    TaskSession:str
    TaskDesc:str
    Time:str 
   
    Type:str
    RelativeBox:str
    RelativeRegion:str
    Properties:str
    property:str
    desc:str
    value:str
    display:str

    def to_po(self):
        po= bizAlarmPO()
        po.uuid=self.UniqueId
        po.alarmType=self.Summary 
        po.taskSession=self.TaskSession
        po.taskDesc=self.TaskDesc 
        po.alarmTime=datetime.fromisoformat(self.Time)
        return po
    def date(self):
        return ""


class HWX_Param:
    """
    惠纬讯设备上传 结构
    """
    m: str  # ": "alarm_event",
    snvr: str  # ": "SNVR-TZSR-SR00-5000",
    vcam: str  # ": "VCAM-TZSH-SR00-5001",
    capture_type: int  # ": 0,
    record_ver: int  # ": 1,
    capture_type: int  # ": 0,
    capture_user: str  # ": "ruleAuto",
    dt_alarm: int  # ": 1700095524875529,
    record_dir: str  # ": "/resources/record/2023/11/16/08/VCAM-TZSH-SR00-5001",
    clip_secs: int  # ": 359,
    jpeg_num: int  # ": 1,
    capture_type: int  # ": 0,
    file_upload_type: int  # ": 1,
    alarm_lat: str  # ":"32.199847N",
    alarm_lng: str  # ":"119.046082E",
    gps_speed: float  # ": 9.754,
    gps_dir: str  # ": 41.060,
    odo_update_st: int  # ": 1700095503227940,
    odo_name: str  # ": "据吴淞口310",
    odo_mileage: float  # ":310.000,
    yz_saildir: int  # ":2,
    scene: str  # ":"sailing",
    break_rules: str  # ": "CHUNBO-v2-5",
    serial: int  # ": 5

    def to_po(self):
        po=bizAlarmPO()
        poa=bizAlarmAttachPO()  # 附加信息
        poa.result=json.dumps({ 
            "m": self.m,"snvr": self.snvr,    
            "vcam": self.vcam ,"capture_type": self.capture_type,
            "record_ver":self.record_ver,
            "capture_type":self.capture_type,
            "capture_user":self.capture_user,
            "dt_alarm":self.dt_alarm,
            "record_dir":self.record_dir,
            "clip_secs":self.clip_secs,
            "jpeg_num":self.jpeg_num,
            "capture_type":self.capture_type,
            "file_upload_type":self.file_upload_type, 
            "odo_update_st":self.odo_update_st,
            "odo_name":self.odo_name,
            "odo_mileage":self.odo_mileage,
            "yz_saildir":self.yz_saildir,
            "scene":self.scene,
            "break_rules":self.break_rules,
            "serial":self.serial,
        }) 
         
        poa.gps=json.dumps({ "alarm_lat": self.alarm_lat, "alarm_lng": self. alarm_lng ,"gps_speed": self. gps_speed ,"gps_dir": self.gps_dir}) 
        po.alarmAttachPO=poa 
        return po
