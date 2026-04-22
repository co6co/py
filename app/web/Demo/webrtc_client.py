import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpreceiver import RemoteStreamTrack
import cv2
from model.apphelp import get_config
from model.rtc_util import create_offer,on_ws_message
from co6co.utils import log
from websockets.legacy.protocol import WebSocketCommonProtocol 
config = get_config()
port = config.get("port")

def handler_track(pc : RTCPeerConnection):
    @pc.on("track")
    async def on_track(track:RemoteStreamTrack):
        print(f"收到轨道: {track.kind}, 轨道ID: {track.id}, 状态: {track.readyState}")
        if track.kind == "video":
            # 检查轨道状态
            if track.readyState == "ended":
                print("轨道已结束")
                return
            # 添加轨道状态变化监听
            @track.on("ended")
            def on_ended():
                print("轨道结束事件触发")
            log.warn("接收帧...")
            while track.readyState != "ended":
                try:  
                    frame = await track.recv()  
                    #print(f"成功收到第一帧: {frame.width}x{frame.height}")
                    if frame is None:
                        break
                    frame = frame.to_ndarray(format="bgr24") 
                    cv2.imshow("Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    log.err("error",e)
            cv2.destroyAllWindows()

def handler_icecandidate(pc:RTCPeerConnection):
    # 设置 ICE 候选监听
    #@pc.on("icecandidate")
    #def on_icecandidate(candidate):
    #    if candidate:
    #        send_ice_candidate(candidate)
    pass
def handler_icecandidateState(pc:RTCPeerConnection): 
    # 等待 ICE 连接状态
    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        state = pc.iceConnectionState
        print(f"ICE 连接状态: {state}")
        if state == "connected":
            print("P2P 连接已建立!")
        elif state == "failed":
            print("连接失败")
            await pc.close()
    pass
async def offer_hander(pc:RTCPeerConnection,ws:WebSocketCommonProtocol):
    # 创建offer
    offer_sdp=await create_offer(pc,config.get("rtsp_url"))
    await ws.send(json.dumps(offer_sdp)) 
    print("等待 answer...")
    answer_dict = json.loads(await ws.recv())
    print("收到 answer",answer_dict.get("type"))
    # 创建answer对象
    answer = RTCSessionDescription(answer_dict.get("sdp"),  answer_dict.get("type"))
    await pc.setRemoteDescription(answer)
     

async def run_client():
    try:
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            pc = RTCPeerConnection() 
            handler_track(pc)
            handler_icecandidateState(pc)
            await offer_hander(pc,ws) 
            await on_ws_message(pc,ws)
        log.warn("退出")
    except Exception as e:
        log.err("clientError",e) 
asyncio.run(run_client())