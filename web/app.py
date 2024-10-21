# websocket 测试
from sanic import Sanic, Websocket
from sanic.request import Request
from sanic.response import text
from co6co.utils import log
import cv2
import numpy as np
from typing import Type
import datetime

from co6co_sanic_ext.model.res.result import Page_Result, Result

import subprocess
app = Sanic(__name__)
'''
app.config.WEBSOCKET_MAX_SIZE = 2 ** 20
app.config.WEBSOCKET_MAX_QUEUE = 32
app.config.WEBSOCKET_READ_LIMIT = 2 ** 16
app.config.WEBSOCKET_WRITE_LIMIT = 2 ** 16
app.config.WEBSOCKET_PING_INTERVAL = 20
app.config.WEBSOCKET_PING_TIMEOUT = 20
'''

'''
@app.websocket("/ws/stream")
async def handler(request, ws: Websocket):
    messgage = "Start"
    while True:
        await ws.send(message)
        message = ws.recv()
        print(message)


@app.websocket('/feed')
async def feed(request, ws):
    while True:
        data = await ws.recv()
        if data is None:
            break
        # 回传数据给客户端
        await ws.send(f"Echo: {data}")

'''


@app.route("/home")
async def feed(request: Request):
    return text("hellow wolrd")


@app.route("/home2")
async def feed2(request: Request):
    return text("hellow 222")


@app.websocket("/ws")
async def ws(request: Request, ws: Websocket):
    async for msg in ws:
        print("收到客户端消息:", msg)
        await ws.send(msg)


@app.websocket('/getStream')
async def handle_websocket2(request,   ws: Websocket):
    try:
        async def generate_video_stream(ws: Websocket, width=640, height=480):
            while True:
                try:
                    # 生成一个随机帧
                    frame = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    now = datetime.datetime.now()
                    cv2.putText(frame, now.strftime('%H:%M:%S'), (0, 50), font, 2, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, now.strftime('%Y-%m-%d'), (100, 0), font, 2, (0, 255, 0), 1, cv2.LINE_AA)

                    # 将帧编码为H264
                    success, encoded_frame = cv2.imencode('.jpg', frame)

                    if success:
                        data = encoded_frame.tobytes()
                        await ws.send(data)
                except Exception as e:
                    # 出错大多为通道关闭
                    log.info("执行出错", e)
                    break
        while True:
            await generate_video_stream(ws)
    except Exception as e:
        log.err("WebSocket error", e)
    finally:
        # 清理资源
        log.warn("关闭websocket.")

# print(app.config)
# Sanic.serve(app)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8085)
