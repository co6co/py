# websocket 测试
from sanic import Sanic, Websocket
from sanic.request import Request
from sanic.response import text

from typing import Type
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

# print(app.config)
# Sanic.serve(app)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8084)
