
import asyncio
import websockets
import base64

import cv2
import numpy as np
import configparser # 解析ini 文件
# pip install websockets
#capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture = cv2.VideoCapture("rtsp://admin:12345@192.168.3.1:554/h264/ch1/sub/av_stream",cv2.CAP_FFMPEG) 
if not capture.isOpened():
    print('no video')
    quit()
ret, frame = capture.read()
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

# Send image to node server
async def send_msg(websocket):
    global ret, frame
    while ret:
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        # img = data.tostring()
        img = data.tobytes()
        img = base64.b64encode(img).decode()
        print(len(img))
        await websocket.send("data:image/jpeg;base64," + img)
        ret, frame = capture.read()


async def main():
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    WS_HOST = cf.get("websocket", "host")
    PORT = cf.get("websocket", "port")
    WS_URL = 'ws://'+WS_HOST+':'+str(PORT)
    print(WS_URL)
    async with websockets.connect(WS_URL) as websocket:
        await send_msg(websocket)

asyncio.get_event_loop().run_until_complete(main())
