from ast import Dict
from aiohttp.log import ws_logger
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack,VideoStreamTrack, RTCIceCandidate
import cv2
from aiortc.contrib.media import MediaPlayer
from websockets.legacy.protocol import WebSocketCommonProtocol
import websockets
import json
import numpy as np
from co6co.utils import log

class VideoStreamWrapper:
    def __init__(self, source):
        # Windows
        # self.player = MediaPlayer(f"video={source}", format="dshow") 
        # Linux
        # self.player = MediaPlayer(f"/dev/video{source}", format="v4l2") 
        # 文件或 RTSP
        self.player = MediaPlayer(source) 
        if not self.player.video:
            raise ValueError(f"无法打开视频源: {source}")
        
        # 获取视频轨道
        self.video = self.player.video
    
    @property
    def track(self):
        return self.video
class VideoStream(VideoStreamTrack):
    """

    "sendrecv"、"sendonly"
    "recvonly"、"inactive"
    """
    def __init__(self, source):
        super().__init__() 
        self.cap = cv2.VideoCapture(source) 
    async def recv(self):
        """
        接收视频帧 需要改造
        """
        r,frame =self.cap.read()
        if r:
            print("发送帧",frame)
            return frame
        print("视频轨道结束")
        return None
class DummyAudioTrack(MediaStreamTrack):
    """
    静音音轨
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.samples = 0
    
    async def recv(self):
        from av import AudioFrame 
        # 创建静音音频帧
        frame = AudioFrame(format='s16', layout='mono', samples=960)
        frame.planes[0].update(np.zeros(960, dtype=np.int16))
        frame.sample_rate = self.sample_rate
        frame.time_base = '1/48000'
        
        self.samples += 960
        frame.pts = self.samples
        
        return frame
 
class BlackVideoTrack(MediaStreamTrack):
    """
    虚拟视频轨道
    黑色视频轨道
    """
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.counter = 0
    
    async def recv(self):
        from av import VideoFrame
        import numpy as np
        
        # 创建黑色视频帧
        width, height = 640, 480
        frame = VideoFrame(width=width, height=height, format='rgb24')
        
        # 填充黑色像素
        data = np.zeros((height, width, 3), dtype=np.uint8)
        for i, plane in enumerate(frame.planes):
            plane.update(data[..., i].tobytes())
        
        frame.pts = self.counter
        self.counter += 3000  # 假设 30fps
        
        return frame

async def create_offer(pc:RTCPeerConnection,  video_source=None):
    """
    发起提议
    """ 
    #if video_source:
    #    pc.addTrack(VideoStream(video_source))  
    #pc.addTrack(DummyAudioTrack())
    #pc.addTrack(BlackVideoTrack())
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer) #将提议设置为本地描述
    local_desc = {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp
    } 
    return local_desc

async def create_answer(pc:RTCPeerConnection, offer:dict  ,video_source=None):

    """
    应答提议
    :param offer: 对端提议 {type:str,sdp:str}
    :param video_source: 视频源
    :return:
    """
    if video_source:
        print("增加视频轨道",video_source)
        #player = MediaPlayer(f"/dev/video{video_source}", format="v4l2")
        pc.addTrack(VideoStreamWrapper(video_source).track) 
        #pc.addTrack(VideoStream(video_source))
        #pc.addTransceiver(BlackVideoTrack()) 
    await pc.setRemoteDescription(RTCSessionDescription(offer.get("sdp"),type=offer.get("type"))) ## 将对端的应答设置为远程描述
 
    answer= await pc.createAnswer() 
    try:
        # 设置本地描述
        #print("设置本地描述",answer)
        await pc.setLocalDescription(answer) 
        #print("成功设置本地描述")
    except Exception as e:
        print(f"设置本地描述失败: {e}")
        # 尝试打印当前状态 
        for transceiver in pc.getTransceivers():
            log.warn(f"Transceiver: {transceiver.direction}, {transceiver.currentDirection}")
        log.err(f"设置本地描述失败: {e}",e)
        raise 
    local_desc = {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp
    } 
    return local_desc

async def on_ws_message(pc:RTCPeerConnection,ws:WebSocketCommonProtocol):
    # 接收消息循环
    while True:
        try:
            message = await ws.recv()
            print(message)  
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket 连接已关闭")
            break
        except Exception as e:
            print(f"接收消息时出错: {e}")
            break
    
    # 清理
    await pc.close()
async def signaling_exchange(pc:RTCPeerConnection, signaling:WebSocketCommonProtocol):
    while True:
        message = await signaling.recv()
        if isinstance(message, RTCSessionDescription):
            await pc.setRemoteDescription(message)
        elif isinstance(message, RTCIceCandidate):
            await pc.addIceCandidate(message)

class WebRTCConnection:
    """
    WebRTC 连接类 
    如果需要可以进行封装
    未进行测试过
    """
    def __init__(self,ws:WebSocketCommonProtocol,peer_id=None) -> None:
        self.ws = ws
        self.peer_id = peer_id
        self.connected = False
        pass
    def set_peer_connection(self, pc:RTCPeerConnection):
        """设置对等连接并监听 ICE 候选"""
        self.pc = pc
        
        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await self.send_ice_candidate(candidate)
            else:
                # ICE 收集完成
                print("ICE 候选收集完成")
                await self.send_message({
                    "type": "ice_complete"
                })
    
    async def send_ice_candidate(self, candidate: RTCIceCandidate):
        """发送 ICE 候选到对端"""
        candidate_data = {
            "candidate": candidate.candidate,
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex
        }
        
        message = {
            "type": "ice_candidate",
            "candidate": candidate_data
        }
        
        if self.peer_id:
            message["to"] = self.peer_id
            
        await self.send_message(message)
    
    async def send_message(self, message):
        """通过 WebSocket 发送消息"""
        try:
            if self.ws and not self.ws.closed:
                await self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"发送消息失败: {e}")
    
    async def receive_messages(self):
        """接收并处理消息"""
        try:
            async for message in self.ws:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket 连接正常关闭")
        except Exception as e:
            print(f"接收消息错误: {e}")
    
    async def handle_message(self, message):
        """处理接收到的消息"""
        data = json.loads(message)
        
        if data["type"] == "ice_candidate":
            await self.handle_remote_ice_candidate(data["candidate"])
        elif data["type"] == "ice_complete":
            print("对端 ICE 收集完成")
        # ... 处理其他消息类型
    
    async def handle_remote_ice_candidate(self, candidate_data):
        """处理接收到的 ICE 候选"""
        if not self.pc:
            print("对等连接未初始化")
            return
            
        candidate = RTCIceCandidate(
            candidate=candidate_data["candidate"],
            sdpMid=candidate_data["sdpMid"],
            sdpMLineIndex=candidate_data["sdpMLineIndex"]
        )
        
        try:
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            print(f"添加 ICE 候选失败: {e}") 