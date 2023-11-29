
from decimal import Decimal


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

class Video_Param:
    """
    盒子上传视频参数
    """
    BoardIp:str 
    BoardId:str
    ip:str
    TaskSession:str
    GBDeviceId:str
    GBTaskChnId:str
    Video:str
    
class Alert_Param:
    """
    盒子上传视频参数
    """
    BoardIp:str
    BoardId:str
    Addition:str
    AlarmId:str
    ImageData:str
    ImageDataLabeled:str
    Media:str
    LocalRawPath:str
    LocalLabeledPath:str
    TaskSession:str
    GBDeviceId:str
    GBTaskChnId:str
    TaskDesc:str
    Time:str
    VideoFile:str
    GPS:str
    available:bool
    kSpeed:Decimal
    nSpeed:Decimal
    latitude:Decimal
    longitude:Decimal
    Result:str
    Type:str
    RelativeBox:str
    RelativeRegion:str
    Properties:str
    property:str
    desc:str
    value:str
    display:str