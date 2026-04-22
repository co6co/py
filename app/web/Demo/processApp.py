 
import signal
import sys 
from sanic import Sanic, Request 
from sanic.response import html 
from model.apphelp import read_file_content, get_file_path 
from co6co.utils import try_except, log 
from model.apphelp import get_config
app = Sanic("RTSP_WebSocket_Proxy")
 
 
  
@app.route('/')
async def index(request:Request):
    """主页面 - 从文件读取HTML"""
    fiel_path = get_file_path('index.html')
    html_content = read_file_content(fiel_path)
    return html(html_content)


def signal_handler(sig, frame):
    """优雅关闭"""
    print("\n[INFO] 收到关闭信号，清理资源...") 
    sys.exit(0)

import os
pgid= os.getpid()
log.succ(f"{pgid}主文件 __name__{__name__}",__file__)
if __name__ == '__main__': 
    log.succ(f"{pgid}main __name__{__name__}" )
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 
    # 运行服务器
    config=get_config()
    port=config.get("port")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        access_log=False,
        auto_reload=False
    )
