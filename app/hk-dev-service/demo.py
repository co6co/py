from services.hik_service import HkService,HkServiceWeb
import os
from co6co.utils import log


service = HkService()
try:
    web=HkServiceWeb("192.168.3.1", "xxx", "xxx")
    log.start_mark("web")
    web.getDeviceInfo() 
    log.end_mark("web") 
finally:
    service.Close()
