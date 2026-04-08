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


async def stream_rtsp_to_ws(ws: WebSocketCommonProtocol, rtsp_url: str, ws_key: str):
    """从RTSP拉流并通过WebSocket转发"""
    try:
        print(f"++++++++++++++++++++开始拉流: {rtsp_url}")
        async for data in rtsService.read_rtsp_stream(rtsp_url, ws_key):
            # async for data in rtsService.exec_ipconfig():
            print("发送数据", len(data))
            isOpen = await check_connection_alive()
            if isOpen:
                print("ws opend...", len(data))
                await ws.send(data)
            else:
                break
    except Exception as e:
        print(f"[ERROR] stream_rtsp_to_ws: {e}")

# @app.listener('before_server_start')
# async def setup_background_tasks(app, loop):
#    """设置后台任务管理器"""
#    app.ctx.background_tasks = {}


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
        task = stream_rtsp_to_ws(ws, rtsp_url, ws_key)
        stream_task = asyncio.ensure_future(task)
        # stream_task = asyncio.create_task(task)   # 直接使用 asyncio.create_task()可能与 Sanic 的任务管理冲突 导致子进程不工作
        # stream_task = app.add_task(task)
        # stream_task = asyncio.ensure_future(task)
        # app.ctx.background_tasks[ws_key] = task
        # 保持连接活跃
        while True:
            message = await ws.recv()
            if message == "close":
                print(f"[INFO] 收到关闭指令")
                stream_task.cancel()
                break
            # 可以处理其他控制消息
            # 例如: 暂停、恢复、改变参数等

    except Exception as e:
        print(f"[INFO] 连接断开: {e}")
    finally:
        # 取消任务并清理
        # task = app.ctx.background_tasks.pop(ws_key, None)
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

        rtsService.stop(ws_key)
        print(f"[INFO] 连接清理完成: {ws_key}")


@app.route('/')
async def index(request):
    """主页面 - 从文件读取HTML"""
    fiel_path = get_file_path('index.html')
    html_content = read_file_content(fiel_path)
    return html(html_content)


def signal_handler(sig, frame):
    """优雅关闭"""
    print("\n[INFO] 收到关闭信号，清理资源...")
    del rtsService
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
