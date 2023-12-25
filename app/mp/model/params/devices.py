

from typing import List


class streamUrl:
    name: str
    url: str


class cameraParam:
    deviceType: int
    innerIp: str
    name: str
    streamUrls: List[streamUrl]
