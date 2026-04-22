import asyncio
import json
import base64
import time
from typing import Set
import subprocess
import websockets
from websockets.server import WebSocketServerProtocol
from model.apphelp import get_config



class SimpleRTSPtoWebSocket:
    """简化的RTSP转WebSocket服务器"""

    def __init__(self, rtsp_url: str, port: int = 8765):
        self.rtsp_url = rtsp_url
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.ffmpeg_process = None

    async def start_ffmpeg(self):
        """启动FFmpeg进程"""
        # 启动FFmpeg，输出MPEG-TS格式
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',  # 使用TCP
            '-i', self.rtsp_url,       # RTSP输入
            '-c:v', 'libx264',         # 视频编码  H.264/AVC​
            '-preset', 'ultrafast',    # 最快编码
            '-tune', 'zerolatency',    # 零延迟
            '-g', '50',                # GOP大小
            '-f', 'mpegts',            # 容器格式
            '-c:a', 'aac',             # 音频编码 AAC​
            '-b:a', '128k',            # 音频比特率
            'pipe:1'                   # 输出到标准输出
        ]

        self.ffmpeg_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        print(f"FFmpeg进程已启动 (PID: {self.ffmpeg_process.pid})")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        self.clients.add(websocket)
        print(f"客户端连接: {websocket.remote_address}")

        try:
            # 发送欢迎消息
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Connected to video stream",
                "timestamp": int(time.time() * 1000)
            }))

            # 读取客户端消息（保持连接）
            async for message in websocket:
                # 可以处理客户端命令
                if message == "ping":
                    await websocket.send("pong")

        except Exception as e:
            print(f"客户端错误: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"客户端断开: {websocket.remote_address}")

    async def broadcast_video(self):
        """广播视频流到所有客户端"""
        if not self.ffmpeg_process or not self.ffmpeg_process.stdout:
            return

        # 发送流信息
        stream_info = {
            "type": "stream_info",
            "codec": "h264",
            "container": "mpegts",
            "timestamp": int(time.time() * 1000)
        }

        for client in self.clients:
            try:
                await client.send(json.dumps(stream_info))
            except:
                pass

        # 读取并广播视频数据
        try:
            while True:
                chunk = await self.ffmpeg_process.stdout.read(4096)
                if not chunk:
                    break

                # 发送数据（使用base64编码）
                message = {
                    "type": "video",
                    "data": base64.b64encode(chunk).decode('utf-8'),
                    "timestamp": int(time.time() * 1000)
                }
                # with open(f'./tmps/video-{int(time.time() * 1000)}.jpeg', 'wb') as f:
                #    f.write(chunk)
                # 发送到所有客户端
                disconnected_clients = []
                for client in self.clients:
                    try:
                        await client.send(json.dumps(message))
                    except:
                        disconnected_clients.append(client)

                # 清理断开连接的客户端
                for client in disconnected_clients:
                    if client in self.clients:
                        self.clients.remove(client)

        except Exception as e:
            print(f"广播视频出错: {e}")

    async def start(self):
        """启动服务器"""
        # 启动FFmpeg
        await self.start_ffmpeg()

        # 启动WebSocket服务器
        async with websockets.serve(
            self.handle_client,
            "0.0.0.0",
            self.port,
            ping_interval=20,
            ping_timeout=40
        ):
            print(f"WebSocket服务器启动在 ws://0.0.0.0:{self.port}")

            # 启动视频广播
            await self.broadcast_video()


async def main_simple():
    """简化版主函数"""
    # RTSP URL示例: rtsp://admin:password@192.168.1.100:554/stream1
    
    config=get_config()
    port=config.get("port")
    rtsp_url=config.get("rtsp_url")
    #import argparse
    #parser = argparse.ArgumentParser(description="RTSP流处理服务器")
    #parser.add_argument("--rtsp_url", type=str, help="RTSP流URL")
    #args = parser.parse_args() 
    server = SimpleRTSPtoWebSocket(rtsp_url, port=port)
    await server.start()


if __name__ == '__main__':
    asyncio.run(main_simple()) 
