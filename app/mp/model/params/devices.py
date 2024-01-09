

from typing import List
from model.pos.biz import bizCameraPO
import json

class streamUrl:
    name: str
    url: str


class cameraParam:
    cameraType: int=None
    innerIp: str=None
    name: str=None
    streamUrls: List[streamUrl] 
    sip: str=None
    channel1_sip: str=None
    channel2_sip: str=None
    channel3_sip: str=None
    channel4_sip: str=None
    channel5_sip: str=None
    siteId: int=None
    talkbackNo: int=None

    def set_po(self, po:bizCameraPO):
        print(self.cameraType)
        po.CameraType=self.cameraType
        po.innerIp=self.innerIp
        po.name=self.name
        
        po.streams = json.dumps(self.streamUrls) 
        po.sip=self.sip
        po.channel1_sip=self.channel1_sip
        po.channel2_sip=self.channel2_sip
        po.channel3_sip=self.channel3_sip
        po.channel4_sip=self.channel4_sip
        po.channel5_sip=self.channel5_sip
        po.siteId=self.siteId
        po.talkbackNo=self.talkbackNo
