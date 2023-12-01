
from decimal import Decimal
from model.pos.right import bizAlarmPO
from datetime import datetime


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