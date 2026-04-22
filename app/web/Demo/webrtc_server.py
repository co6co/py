import asyncio
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import cv2,json
from model.apphelp import get_config
from model.rtc_util import VideoStream, create_answer,on_ws_message
from websockets.legacy.protocol import WebSocketCommonProtocol 
  
async def signaling_server(ws:WebSocketCommonProtocol , path:str ):
    print("new connection...",path,type(ws))
    config = get_config() 
    pc = RTCPeerConnection()  
    # 设置 ICE 候选监听
    #@pc.on("icecandidate")
    #def on_icecandidate(candidate):
    #    if candidate:
    #        send_ice_candidate(candidate)
    #pc.addTrack(VideoStream(config.get('rtsp_url'))) 
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange(): 
        state = pc.iceConnectionState
        if state in ["connected", "completed"]:
            # 连接成功
            print("连接成功")
            pass
        elif state in ["failed", "disconnected"]:
            # 连接失败
            print("连接失败")
            await pc.close()
            pass 
    offer = json.loads(await ws.recv()) 
    print("收到 offer",offer,type(offer)) 
    answer_sdp=await create_answer(pc,offer,config.get('rtsp_url'))
    await ws.send(json.dumps(answer_sdp))
    await on_ws_message(pc,ws)


def test(rtsp_url):
    stream=VideoStream(rtsp_url)
    async def get_frame():
        while True:
            frame = await stream.recv() 
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
    loop_event.create_task(get_frame())


if __name__ == "__main__":
    config = get_config()
    port = config.get("port")  
    loop_event = asyncio.get_event_loop()
   
    start_server = websockets.serve(signaling_server, "localhost", port)
    print (f"running on localhost port {port}...")
    loop_event.run_until_complete(start_server)
    loop_event.run_forever()
