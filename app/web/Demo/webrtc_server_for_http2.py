import asyncio
import json
import os
import logging
import signal
import sys
from datetime import datetime
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCIceCandidate,
)
from aiortc.contrib.media import MediaPlayer
import numpy as np
import cv2
import numpy as np
from model.rtc_util import VideoStreamWrapper, DummyAudioTrack
from co6co.utils import log
from typing import Dict
from model.aiohttp_server import aio_http_server_base

log_dir = "./dist/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(log_dir, "webrtc_server2.log"), encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)


class WebRTCStreamServer(aio_http_server_base):
    def __init__(self, port=8766):
        self.clients: Dict[str, web.WebSocketResponse] = {}
        self.streams = {}
        self.rtsp_url = None
        self.subscriber_pcs = {}
        super().__init__(port)

    async def print_info(self):
        w, h = ("ws", "http")
        if self.ssl:
            w, h = ("wss", "https")

        logger.info(f"WebSocket 端点: {w}://localhost:{self.port}/ws")
        logger.info(f"HTTP 静态文件: {h}://localhost:{self.port}/")
        logger.info(f"状态端点: {h}://localhost:{self.port}/status")

    async def index(request):
        # 构建 index.html 的路径
        index_path = os.path.join(
            os.path.dirname(__file__), "pages", "webrtc_client2.html"
        )
        # 读取文件内容并返回响应
        with open(index_path, "rb") as f:
            content = f.read()
        return web.Response(body=content, content_type="text/html")

    async def route_handler(self, app: web.Application):
        app.add_routes(
            [
                web.get("/ws", self.websocket_handler),
                web.get("/status", self.status_handler),
                web.get("/api/streams", self.streams_handler),
                web.post("/api/rtsp", self.set_rtsp_handler),
                web.static("/", os.path.join(os.path.dirname(__file__), "pages")),
                web.get("/", self.index),  # 添加根路径的处理器
            ]
        )
        await self.print_info()

    async def status_handler(self, request):
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "clients_count": len(self.clients),
            "streams_count": len(self.streams),
            "streams": list(self.streams.keys()),
            "rtsp_url": self.rtsp_url if self.rtsp_url else "未设置",
        }
        return web.json_response(status)

    async def streams_handler(self, request):
        streams_info = []
        for stream_id, info in self.streams.items():
            streams_info.append(
                {
                    "stream_id": stream_id,
                    "type": info.get("type"),
                    "source": info.get("source", "unknown"),
                    "subscribers": info.get("subscribers_count", 0),
                }
            )
        return web.json_response(
            {"total_streams": len(streams_info), "streams": streams_info}
        )

    async def set_rtsp_handler(self, request):
        data = await request.json()
        self.rtsp_url = data.get("rtsp_url")
        logger.info(f"RTSP URL 设置为: {self.rtsp_url}")
        return web.json_response({"status": "success", "rtsp_url": self.rtsp_url})

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("新的客户端连接")
        peer_id = None

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        peer_id = await self.handle_message(ws, data, peer_id)
                    except json.JSONDecodeError as e:
                        logger.error(f"消息解析错误：{e}")
                    except Exception as e:
                        logger.error(f"处理消息时发生错误：{e}")
                elif msg.type == web.WSMsgType.BINARY:
                    logger.warning("收到二进制消息，当前不支持")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket 错误：{ws.exception()}")
        except Exception as e:
            logger.error(f"WebSocket 连接异常：{e}")
        finally:
            if peer_id:
                await self.handle_disconnect(ws, peer_id)
        return ws

    async def handle_message(self, ws: web.WebSocketResponse, data: dict, peer_id):
        msg_type = data.get("type")

        if msg_type == "register":
            return await self.handle_register(ws, data)
        elif msg_type == "publish":
            await self.handle_publish(ws, data)
        elif msg_type == "subscribe":
            await self.handle_subscribe(ws, data)
        elif msg_type == "offer":
            await self.handle_offer(ws, data)
        elif msg_type == "answer":
            await self.handle_answer(data)
        elif msg_type == "ice-candidate":
            await self.handle_ice_candidate(data)
        elif msg_type == "unpublish":
            await self.handle_unpublish(data)
        elif msg_type == "unsubscribe":
            await self.handle_unsubscribe(data)
        elif msg_type == "start-rtsp":
            await self.handle_start_rtsp(ws, data)
        else:
            logger.warning(f"未知消息类型：{msg_type}")
        return peer_id

    async def handle_register(self, ws: web.WebSocketResponse, data: dict):
        peer_id = data.get("peerId")
        if peer_id in self.clients:
            await self.send_to(ws, {"type": "error", "message": "Peer ID 已存在"})
            return peer_id

        self.clients[peer_id] = {"ws": ws}
        await self.send_to(ws, {"type": "registered", "peerId": peer_id})
        logger.info(f"客户端注册成功：{peer_id}")
        await self.send_stream_info(ws, self.get_stream_list())
        return peer_id

    async def handle_publish(self, ws, data: dict):
        peer_id = data.get("from")
        stream_id = data.get("streamId", f"stream_{peer_id}")
        source_type = data.get("sourceType", "camera")
        source = data.get("source", "camera")

        if stream_id in self.streams:
            await self.send_to(ws, {"type": "error", "message": "流已存在"})
            return

        self.streams[stream_id] = {
            "publisher": peer_id,
            "type": source_type,
            "source": source,
            "subscribers": set(),
            "subscribers_count": 0,
        }

        await self.send_to(ws, {"type": "publish-success", "streamId": stream_id})
        await self.broadcast_stream_update()
        logger.info(f"流发布成功：{stream_id} by {peer_id}, 类型: {source_type}")

    async def handle_subscribe(self, ws: web.WebSocketResponse, data: dict):
        peer_id = data.get("from")
        stream_id = data.get("streamId")

        if stream_id not in self.streams:
            await self.send_to(ws, {"type": "error", "message": "流不存在"})
            return

        stream_info = self.streams[stream_id]
        stream_info["subscribers"].add(peer_id)
        stream_info["subscribers_count"] = len(stream_info["subscribers"])

        await self.send_to(ws, {"type": "subscribe-success", "streamId": stream_id})
        await self.broadcast_stream_update()

        if stream_info.get("type") == "rtsp":
            await self.create_rtsp_peer_connection(peer_id, stream_id, ws)
        else:
            publisher_id = stream_info.get("publisher")
            if publisher_id in self.clients:
                publisher_ws = self.clients[publisher_id]["ws"]
                await self.send_to(
                    publisher_ws,
                    {"type": "new-subscriber", "from": peer_id, "streamId": stream_id},
                )

        logger.info(f"{peer_id} 订阅了流 {stream_id}")

    async def create_rtsp_peer_connection(
        self, peer_id, stream_id, ws: web.WebSocketResponse
    ):
        try:
            stream_info = self.streams.get(stream_id)
            if not stream_info:
                logger.error(f"流不存在: {stream_id}")
                return

            pc = RTCPeerConnection(
                RTCConfiguration(
                    # /coturn/coturn
                    iceServers=[
                        # RTCIceServer(urls="stun:stun.l.google.com:19302"),
                        # RTCIceServer(urls="stun:stun1.l.google.com:19302")
                        RTCIceServer(urls="stun:stun.qq.com:3478"),
                        RTCIceServer(urls="stun:stun.aliyun.com:3478"),
                        RTCIceServer(urls="stun:stun.miwifi.com:3478"),
                    ]
                )
            )
            logger.info(
                f"为 {peer_id} 创建 PeerConnection，视频源: {stream_info['source']}"
            )

            player = MediaPlayer(stream_info["source"])
            if player.video:
                pc.addTrack(player.video)
                logger.info("已添加视频轨道")
            else:
                logger.error("无法获取视频轨道")
                await pc.close()
                return
            if player.audio:
                logger.info("增加rtsp音轨")
                pc.addTrack(player.audio)
            else:
                pc.addTrack(DummyAudioTrack())
                logger.info("已添加静音音频轨道")

            pc.on(
                "icecandidate",
                lambda e: asyncio.ensure_future(
                    self.send_ice_candidate(e, peer_id, ws)
                ),
            )

            pc.on(
                "connectionstatechange",
                lambda: asyncio.ensure_future(
                    self.on_connection_state_change(pc, peer_id, stream_id)
                ),
            )

            self.subscriber_pcs[peer_id] = pc
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            await self.send_to(
                ws,
                {
                    "type": "offer",
                    "sdp": offer.sdp,
                    "from": "server",
                    "streamId": stream_id,
                },
            )
            logger.info(f"已向 {peer_id} 发送 RTSP 流 Offer")

        except Exception as e:
            logger.error(f"创建 RTSP PeerConnection 失败: {e}", exc_info=True)
            if peer_id in self.subscriber_pcs:
                await self.subscriber_pcs[peer_id].close()
                del self.subscriber_pcs[peer_id]

    async def send_ice_candidate(self, event, peer_id, ws: web.WebSocketResponse):
        log.warn("send_ice_candidate", type(event))
        if event.candidate and peer_id in self.clients:
            candidate_dict = {
                "candidate": event.candidate.candidate,
                "sdpMid": event.candidate.sdpMid,
                "sdpMLineIndex": event.candidate.sdpMLineIndex,
            }
            await self.send_to(
                ws,
                {
                    "type": "ice-candidate",
                    "candidate": candidate_dict,
                    "from": "server",
                },
            )

    async def on_connection_state_change(
        self, pc: RTCPeerConnection, peer_id: str, stream_id: str
    ):

        state = pc.connectionState
        logger.info(f"PeerConnection 状态变化: {peer_id} -> {state}")

        if state == "failed" or state == "disconnected":
            if peer_id in self.subscriber_pcs:
                await self.subscriber_pcs[peer_id].close()
                del self.subscriber_pcs[peer_id]

            if stream_id in self.streams:
                self.streams[stream_id]["subscribers"].discard(peer_id)
                self.streams[stream_id]["subscribers_count"] = len(
                    self.streams[stream_id]["subscribers"]
                )
            await self.broadcast_stream_update()

    async def handle_start_rtsp(self, ws: web.WebSocketResponse, data: dict):
        peer_id = data.get("from")
        rtsp_url = data.get("rtspUrl", self.rtsp_url)

        if not rtsp_url:
            await self.send_to(ws, {"type": "error", "message": "RTSP URL 未设置"})
            return

        stream_id = f"rtsp_stream_{peer_id}"

        if stream_id in self.streams:
            await self.send_to(ws, {"type": "error", "message": "RTSP 流已存在"})
            return

        self.streams[stream_id] = {
            "publisher": peer_id,
            "type": "rtsp",
            "source": rtsp_url,
            "subscribers": set(),
            "subscribers_count": 0,
            "player": None,
        }

        await self.send_to(ws, {"type": "rtsp-started", "streamId": stream_id})
        await self.broadcast_stream_update()
        logger.info(f"RTSP 流发布成功：{stream_id}, URL: {rtsp_url}")

    async def handle_unpublish(self, data: dict):
        stream_id = data.get("streamId")
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            if stream.get("player"):
                stream["player"].stop()
            del self.streams[stream_id]
            await self.broadcast_stream_update()
            logger.info(f"流已取消发布：{stream_id}")

    async def handle_unsubscribe(self, data: dict):
        peer_id = data.get("from")
        stream_id = data.get("streamId")

        if peer_id in self.subscriber_pcs:
            await self.subscriber_pcs[peer_id].close()
            del self.subscriber_pcs[peer_id]

        if stream_id in self.streams:
            self.streams[stream_id]["subscribers"].discard(peer_id)
            self.streams[stream_id]["subscribers_count"] = len(
                self.streams[stream_id]["subscribers"]
            )

        await self.broadcast_stream_update()
        logger.info(f"{peer_id} 取消订阅流 {stream_id}")

    async def handle_offer(self, ws: web.WebSocketResponse, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        sdp = data.get("sdp")
        stream_id = data.get("streamId")

        if to_peer == "server":
            if from_peer in self.subscriber_pcs:
                pc = self.subscriber_pcs[from_peer]
                try:
                    await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await self.send_to(
                        ws, {"type": "answer", "sdp": answer.sdp, "from": "server"}
                    )
                    logger.info(f"已向 {from_peer} 发送 Answer")
                except Exception as e:
                    logger.error(f"处理服务器 Offer 失败: {e}")
            return

        if to_peer not in self.clients:
            logger.error(f"目标客户端不存在: {to_peer}")
            return

        await self.forward_to_peer(
            to_peer,
            {"type": "offer", "sdp": sdp, "from": from_peer, "streamId": stream_id},
        )

    async def handle_answer(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        sdp = data.get("sdp")

        if to_peer == "server":
            if from_peer in self.subscriber_pcs:
                pc = self.subscriber_pcs[from_peer]
                try:
                    await pc.setRemoteDescription(RTCSessionDescription(sdp, "answer"))
                    logger.info(f"已设置远程描述: {from_peer}")
                except Exception as e:
                    logger.error(f"处理服务器 Answer 失败: {e}", exc_info=True)
            return

        await self.forward_to_peer(
            to_peer, {"type": "answer", "sdp": sdp, "from": from_peer}
        )

    async def handle_ice_candidate(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        candidate_data = data.get("candidate")

        if to_peer == "server":
            if from_peer in self.subscriber_pcs:
                pc = self.subscriber_pcs[from_peer]
                try:
                    from aioice import Candidate as IceCandidate

                    ice_candidate = IceCandidate.from_sdp(
                        candidate_data.get("candidate")
                    )
                    aiortc_candidate = RTCIceCandidate(
                        component=ice_candidate.component,
                        foundation=ice_candidate.foundation,
                        ip=ice_candidate.host,
                        port=ice_candidate.port,
                        priority=ice_candidate.priority,
                        protocol=ice_candidate.transport,
                        type=ice_candidate.type,
                        relatedAddress=ice_candidate.related_address,
                        relatedPort=ice_candidate.related_port,
                        sdpMid=candidate_data.get("sdpMid"),
                        sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                    )
                    await pc.addIceCandidate(aiortc_candidate)
                    logger.info(f"已添加 ICE 候选: {from_peer}")
                except Exception as e:
                    logger.error(f"添加 ICE 候选失败: {e}", exc_info=True)
            return

        await self.forward_to_peer(
            to_peer,
            {"type": "ice-candidate", "candidate": candidate_data, "from": from_peer},
        )

    async def handle_disconnect(self, ws: web.WebSocketResponse, peer_id: str):
        if peer_id in self.clients:
            del self.clients[peer_id]

        if peer_id in self.subscriber_pcs:
            await self.subscriber_pcs[peer_id].close()
            del self.subscriber_pcs[peer_id]

        for stream_id in list(self.streams.keys()):
            stream = self.streams[stream_id]
            if stream["publisher"] == peer_id:
                if stream.get("player"):
                    stream["player"].stop()
                del self.streams[stream_id]
            else:
                stream["subscribers"].discard(peer_id)
                stream["subscribers_count"] = len(stream["subscribers"])

        await self.broadcast_stream_update()
        logger.info(f"客户端断开连接：{peer_id}")

    def get_stream_list(self):
        """获取流列表"""
        return [
            {
                "streamId": stream_id,
                "publisher": info["publisher"],
                "type": info.get("type", "unknown"),
                "subscribers": info["subscribers_count"],
            }
            for stream_id, info in self.streams.items()
        ]

    async def send_stream_info(self, ws, streams_list):
        """给客户端发送流信息"""
        await self.send_to(ws, {"type": "streams-update", "streams": streams_list})

    async def broadcast_stream_update(self):
        """
        广播流信息
        """
        streams_list = self.get_stream_list()
        for client_info in self.clients.values():
            await self.send_stream_info(client_info["ws"], streams_list)

    async def forward_to_peer(self, peer_id: str, message: dict):
        """
        发送消息给客户端
        """
        if peer_id in self.clients:
            await self.send_to(self.clients[peer_id]["ws"], message)

    async def send_to(self, ws: web.WebSocketResponse, message: dict):
        """
        发送消息给客户端
        """
        try:
            await ws.send_str(json.dumps(message))
        except Exception as e:
            logger.error(f"发送消息错误: {e}")


if __name__ == "__main__":
    rtsp_url = sys.argv[1] if len(sys.argv) > 1 else None
    from model.apphelp import get_config

    config = get_config()
    port = config.get("port")
    server = WebRTCStreamServer(port)
    ssl = config.get("ssl")
    if ssl:
        server.use_ssl(config.get("cert"), config.get("key"))
    asyncio.run(server.start())
