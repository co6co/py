from utils.cvUtils import screenshot,showImage,getTempFileName
from co6co.utils import log
url="wss://stream.jshwx.com.cn:8441/flv_ws?device=gb34020000000010363001?type=rt.flv"
#url="ws://localhost:8899"
import asyncio
#showImage(url)

data=asyncio.run(screenshot(url,useBytes=True)) 
log.warn(data)  