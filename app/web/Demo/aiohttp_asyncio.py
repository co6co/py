#!/usr/bin/env python3
"""
完整的 WebRTC RTSP 流媒体服务器
支持：
1. 多路 RTSP 流
2. 自动重连
3. 连接管理
4. 状态监控
"""
import asyncio
import json
import logging
import os
from typing import Dict, List
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRelay
from dataclasses import dataclass
import uuid
from datetime import datetime
from aiohttp.web_request import Request
from model.apphelp import read_file_content, get_file_path,get_config
from co6co.utils import log
# 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webrtc-server")
config=get_config()
devict=config.get("devices")

# RTSP 流配置
STREAMS = config["devices"]  

@dataclass
class ClientInfo:
    """客户端信息"""
    pc_id: str
    pc: RTCPeerConnection
    ws: web.WebSocketResponse
    camera_id: str
    connected_at: datetime
    remote_addr: str


class WebRTCStreamManager:
    """WebRTC 流管理器"""

    def __init__(self):
        self.clients: Dict[str, ClientInfo] = {}
        self.relay = MediaRelay()
        self.player_cache: Dict[str, MediaPlayer] = {}  # 缓存播放器，避免重复创建

    async def get_camera_track(self, camera_id: str):
        """获取摄像头的视频轨道"""
        if camera_id not in STREAMS:
            raise ValueError(f"Camera {camera_id} not found")

        # 如果播放器已存在，复用
        if camera_id in self.player_cache:
            player = self.player_cache[camera_id]
        else:
            # 创建新的播放器
            log.warn(f"创建新的播放器: {camera_id}")
            rtsp_url = STREAMS[camera_id]
            try:
                player = MediaPlayer(
                    rtsp_url,
                    format='rtsp',
                    options={
                        'rtsp_transport': 'tcp',
                        'fflags': 'nobuffer',
                        'flags': 'low_delay',
                        'timeout': '5000000',  # 5秒超时
                        'stimeout': '5000000',  # TCP超时
                    }
                )
                self.player_cache[camera_id] = player
            except Exception as e:
                logger.error(f"Error creating MediaPlayer for {camera_id}: {e}")
                raise

        if player.video:
            return self.relay.subscribe(player.video)
        return None

    async def cleanup_client(self, pc_id: str):
        """清理客户端资源"""
        if pc_id in self.clients:
            client = self.clients[pc_id]
            try:
                await client.pc.close()
            except:
                pass
            del self.clients[pc_id]
            logger.info(f"Cleaned up client {pc_id}")

            # 如果没有客户端使用某个摄像头，清理播放器
            camera_id = client.camera_id
            if camera_id in self.player_cache:
                other_clients = [
                    c for c in self.clients.values()
                    if c.camera_id == camera_id
                ]
                if not other_clients:
                    del self.player_cache[camera_id]
                    logger.info(f"Cleaned up camera player: {camera_id}")

    def get_stats(self) -> Dict:
        """获取服务器统计信息"""
        return {
            "total_clients": len(self.clients),
            "active_cameras": len(self.player_cache),
            "clients": [
                {
                    "id": client.pc_id,
                    "camera": client.camera_id,
                    "connected_at": client.connected_at.isoformat(),
                    "connection_state": client.pc.connectionState,
                    "remote_addr": client.remote_addr
                }
                for client in self.clients.values()
            ]
        }


# 创建管理器实例
stream_manager = WebRTCStreamManager()

# 路由处理器


async def index(request: Request):
    """主页面 - 从文件读取HTML"""

    name = request.query.get('name')
    if not name:
        name = "webrtc0"
    fiel_path = get_file_path(f'{name}.html')
    html_content = read_file_content(fiel_path)
    return web.Response(text=html_content, content_type="text/html")


async def cameras(request):
    """获取摄像头列表"""
    return web.json_response({
        "cameras": [
            {"id": cam_id, "url": url}
            for cam_id, url in STREAMS.items()
        ]
    })


async def stats(request):
    """获取服务器状态"""
    return web.json_response(stream_manager.get_stats())


async def websocket_handler(request):
    """WebSocket 连接处理器"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc_id = str(uuid.uuid4())
    client_ip = request.remote

    logger.info(f"新连接: {pc_id} from {client_ip}")

    try:
        # 接收客户端消息
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data.get("type") == "offer":
                    # 创建 WebRTC 连接
                    camera_id = data.get("camera_id", "camera1")

                    pc = RTCPeerConnection()
                    client_info = ClientInfo(
                        pc_id=pc_id,
                        pc=pc,
                        ws=ws,
                        camera_id=camera_id,
                        connected_at=datetime.now(),
                        remote_addr=client_ip
                    )

                    # 存储客户端信息
                    stream_manager.clients[pc_id] = client_info

                    # 获取视频轨道
                    try:
                        log.warn(f"获取视频轨道 {camera_id}")
                        video_track = await stream_manager.get_camera_track(camera_id)
                        if video_track:
                            pc.addTrack(video_track)
                            logger.info(f"Added video track for {pc_id} (camera: {camera_id})")
                        else:
                            raise Exception("Failed to get video track")
                    except Exception as e:
                        log.warn(f"获取视频轨道 Failed to get camera track: {e}")
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": f"Failed to connect to camera: {str(e)}"
                        }))
                        continue

                    # 设置远程描述
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type="offer")
                    )

                    # 创建并发送 Answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)

                    await ws.send_str(json.dumps({
                        "type": "answer",
                        "sdp": answer.sdp
                    }))

                    logger.info(f"Sent answer to {pc_id}")

                elif data.get("type") == "candidate":
                    # 处理 ICE 候选
                    if pc_id in stream_manager.clients:
                        pc = stream_manager.clients[pc_id].pc
                        candidate_data = data.get("candidate")
                        if candidate_data:
                            try:
                                candidate = RTCIceCandidate(
                                    candidate_data["candidate"],
                                    candidate_data["sdpMid"],
                                    candidate_data["sdpMLineIndex"]
                                )
                                await pc.addIceCandidate(candidate)
                            except Exception as e:
                                logger.error(f"Error adding ICE candidate: {e}")

                elif data.get("type") == "bye":
                    # 客户端断开
                    break

                elif data.get("type") == "ping":
                    # 心跳检测
                    await ws.send_str(json.dumps({"type": "pong"}))

    except Exception as e:
        logger.error(f"WebSocket error for {pc_id}: {e}")

    finally:
        # 清理资源
        await stream_manager.cleanup_client(pc_id)
        logger.info(f"Client {pc_id} disconnected")

    return ws


async def health_check(request):
    """健康检查"""
    return web.Response(text="OK")


async def on_shutdown(app):
    """服务器关闭时清理所有连接"""
    logger.info("Server shutting down, cleaning up connections...")

    # 清理所有客户端
    cleanup_tasks = []
    for pc_id in list(stream_manager.clients.keys()):
        cleanup_tasks.append(stream_manager.cleanup_client(pc_id))

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks)

    logger.info("Cleanup completed")


def create_app():
    """创建应用实例"""
    app = web.Application()
    # 添加路由
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static", "./static")  # 静态文件
    app.router.add_get("/cameras", cameras)
    app.router.add_get("/stats", stats)
    app.router.add_get("/health", health_check)

    # 配置 CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        )
    })
    for route in app.router.routes():
        cors.add(route)

    # 模板目录
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    app.on_shutdown.append(on_shutdown)

    return app


if __name__ == "__main__":
    app = create_app()

    # 启动服务器
    print("=" * 50)
    print("WebRTC RTSP 流媒体服务器")
    print(f"可用摄像头: {list(STREAMS.keys())}")
    # print(f"服务器地址: http://0.0.0.0:8080")
    print("=" * 50)

    web.run_app(
        app,
        host="0.0.0.0",
        port=config.get("port"),
        access_log=None,  # 禁用访问日志
        shutdown_timeout=5
    )
