# signal_server.py
from aiohttp.web_request import Request
import asyncio
import json
import logging
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRelay
import uuid

from model.apphelp import read_file_content, get_file_path,get_config
config=get_config()
url=config.get("rtsp_url")
# 配置
RTSP_URL = url

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-signal")

# 存储PeerConnection
pcs = {}
class WebRTCStreamer:
    def __init__(self):
        self.relay = MediaRelay() 
    async def get_video_track(self):
        """获取视频轨道"""
        # 使用MediaPlayer播放RTSP流
        player = MediaPlayer(
            RTSP_URL,
            format='rtsp',
            options={
                'rtsp_transport': 'tcp',
                'fflags': 'nobuffer',
                'flags': 'low_delay'
            }
        )

        if player.video:
            return self.relay.subscribe(player.video)
        return None


async def index(request: Request):
    """主页面 - 从文件读取HTML"""

    name = request.query.get('name')
    if not name:
        name = "webrtc"
    fiel_path = get_file_path(f'{name}.html')
    html_content = read_file_content(fiel_path)
    return web.Response(text=html_content, content_type="text/html")


async def websocket_handler(request):
    """WebSocket处理"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc_id = str(uuid.uuid4())
    pc = None
    logger.info(f"New WebSocket connection: {pc_id}")

    streamer = WebRTCStreamer()

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)

            if data["type"] == "offer":
                # 创建PeerConnection
                pc = RTCPeerConnection()
                pcs[pc_id] = pc

                # 添加视频轨道
                video_track = await streamer.get_video_track()
                if video_track:
                    pc.addTrack(video_track)

                # 处理Offer
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                )

                # 创建Answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # 发送Answer
                await ws.send_str(json.dumps({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                }))

            elif data["type"] == "candidate":
                if pc and data.get("candidate"):
                    candidate = RTCIceCandidate(
                        data["candidate"]["candidate"],
                        data["candidate"]["sdpMid"],
                        data["candidate"]["sdpMLineIndex"]
                    )
                    await pc.addIceCandidate(candidate)

            elif data["type"] == "bye":
                break

    # 清理
    if pc:
        await pc.close()
        if pc_id in pcs:
            del pcs[pc_id]

    return ws


async def on_shutdown(app):
    """关闭应用"""
    for pc in pcs.values():
        await pc.close()

if __name__ == "__main__":
    app = web.Application()
    # 路由
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static("/static", "./static")  # 静态文件
    # 配置CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # 为所有路由启用CORS
    for route in list(app.router.routes()):
        cors.add(route)

    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=config.get("port"))
