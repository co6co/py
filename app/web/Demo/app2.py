# server.py
import asyncio
import subprocess
import signal
import sys
import os
import datetime
from sanic import Sanic, Request
# from sanic.worker import GatheredTask
from sanic.response import html
# from sanic.websocket import WebSocketCommonProtocol
from websockets.legacy.protocol import WebSocketCommonProtocol
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
from model.services import RTSPService
from model.apphelp import read_file_content, get_file_path
from model.utils import get_client_ip, check_connection_alive 
from co6co.utils import try_except, log
from co6co.task.thread import TaskManage,create_event_loop
import threading

app = Sanic("RTSP_WebSocket_Proxy")

app.static('/static', './static')  # 静态文件目录
app.config.WEBSOCKET_MAX_SIZE = 2 ** 20
app.config.WEBSOCKET_MAX_QUEUE = 32
app.config.WEBSOCKET_READ_LIMIT = 2 ** 16
app.config.WEBSOCKET_WRITE_LIMIT = 2 ** 16
app.config.WEBSOCKET_PING_INTERVAL = 20
app.config.WEBSOCKET_PING_TIMEOUT = 20

# 存储活跃的连接和进程


rtsService = RTSPService()
LOOP=create_event_loop()

async def stream_rtsp_to_ws( rtsp_url: str, ws_key: str,ws: WebSocketCommonProtocol=None):
    """从RTSP拉流并通过WebSocket转发"""
    try:
        print(f"++++++++++++++++++++开始拉流: {rtsp_url}")
        async for data in rtsService.read_rtsp_stream(rtsp_url, ws_key): 
            log.succ(f"发送数据: {len(data)}") 
            isOpen = await check_connection_alive(ws)
            if isOpen: 
                await ws.send(data)
            else:
                break
    except Exception as e:
        log.err("error stream_rtsp_to_ws",e)
        print(f"[ERROR] stream_rtsp_to_ws: {e}")

def demo(LOOP:asyncio.AbstractEventLoop ,  rtsp_url, ws_key,ws: WebSocketCommonProtocol=None):
    #新线程里没有自动创建的 event loop 
    #LOOP=asyncio.get_event_loop() 
    LOOP.create_task(stream_rtsp_to_ws(  rtsp_url, ws_key,ws)) 
    LOOP.run_forever()  # 必须启动事件循环
    pass
def createThread(ws: WebSocketCommonProtocol,  rtsp_url, ws_key): 
    thread=threading.Thread(target=demo,args=(LOOP,  rtsp_url, ws_key,ws,) )
    thread.start()
    return LOOP

 
@app.websocket('/ws/stream')
async def websocket_stream(request, ws: WebSocketCommonProtocol):
    """WebSocket流端点"""
    rtsp_url = request.args.get('url')
    print(rtsp_url)
    if not rtsp_url:
        await ws.close(reason="需要RTSP URL参数")
        return

    # 创建唯一标识符
    import uuid
    ws_key = str(uuid.uuid4()) 
    try:
        # 启动流任务
       
        #log.start_mark("demo 线程")
        #thread_task=createThread(ws, rtsp_url, ws_key)
        #log.end_mark("demo 线程")
        thread_task = TaskManage(event_loop=LOOP,closeEventLoop= False)
        thread_task.runTask(stream_rtsp_to_ws,lambda x:log.info(f"stream_rtsp_to_ws: {x}"), rtsp_url, ws_key,ws)
        while True:
            log.warn("准备接受websocket 消息...")
            message = await ws.recv()
            log.warn("接受websocket 消息.")
            if message == "close": 
                thread_task.stop()
                break
            else:
                print(f"[INFO] 收到消息: {message}")
            # 可以处理其他控制消息
            # 例如: 暂停、恢复、改变参数等

    except Exception as e:
        print(f"[INFO] 连接断开: {e}")
    finally: 
        thread_task.stop()
        try:
            #thread_task.close()
            pass
        except asyncio.CancelledError:
            pass 
  
@app.route('/')
async def index(request):
    """主页面 - 从文件读取HTML"""
    fiel_path = get_file_path('index.html')
    html_content = read_file_content(fiel_path)
    return html(html_content)


def signal_handler(sig, frame):
    """优雅关闭"""
    print("\n[INFO] 收到关闭信号，清理资源...") 
    sys.exit(0)


if __name__ == '__main__':

    # async def test():
    #    async for data in read_rtsp_stream("rtsp://admin:lanbo12345@192.168.3.1/media/video1","test_key"):
    #        print(data)
    # asyncio.run(test())
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 创建静态目录
    import os
    os.makedirs('assets', exist_ok=True)

    # 运行服务器
    app.run(
        host="0.0.0.0",
        port=8801,
        debug=False,
        access_log=False,
        auto_reload=False
    )
