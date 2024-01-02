import asyncio
import signal
from websockets.server import serve
import http

import configparser # 解析ini 文件
 
client={}
async def health_check(path, request_headers):
    if path == "/healthz":
        return http.HTTPStatus.OK, [], b"OK\n"
#https://blog.csdn.net/qq_41375318/article/details/131464865
async def hands(websocket):
    while True:
        recv_content=await websocket.recv()
        if recv_content =="":
            print("connected success")
            await websocket.send("success")
        else:
            await websocket.send("connected fail")
async def bizService(websocket):
    while True:
        async for message in websocket:
            #print(message) 
            await websocket.send(message)




async def service(websocket,path:str): 
    print(f"path:{path}...")
    await hands(websocket)
    await bizService(websocket)
     



async def server():
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    WS_HOST = cf.get("websocket", "host")
    PORT = cf.get("websocket", "port")

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None) 

    async with serve(service, WS_HOST, PORT,process_request=health_check):
        print("client...")
        await stop #asyncio.Future()  # run forever
        print("client...2")

asyncio.run(server())