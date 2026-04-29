import asyncio
import json
import os
import logging
from datetime import datetime
from aiohttp import web
import ssl
from websockets.server import WebSocketServerProtocol 

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./dist/logs/webrtc_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SignalingServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients = {}  # peerId -> {ws, roomId}
        self.rooms = {}    # roomId -> Set of peerIds
        self.websocket = None
        self.ssl=False
    def use_ssl(self, cert, key):
        self.cert = cert
        self.key = key
        self.ssl = True
    def create_ssl_context(self):
         # 配置 SSL 上下文
        if not self.ssl:
            return None
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(self.cert, self.key )
        return ssl_context

    async def print_info(self):
        http="http"
        ws="ws"
        if self.ssl:
            http="https"
            ws="wss"  
        logger.info(f"WebSocket 端点: {ws}://localhost:{self.port}/ws")
        logger.info(f"状态端点: {http}://localhost:{self.port}/status")
        logger.info(f"房间管理已启用")
         
    async def route_handler(self,app:web.Application):
        app.add_routes([
            web.get('/ws', self.websocket_handler),
            web.get('/status', self.status_handler),
            web.get('/api/rooms', self.rooms_handler),
            web.static('/', os.path.join(os.path.dirname(__file__), 'pages')),
        ])

    async def start(self):
        app = web.Application() 
        await self.route_handler(app) 
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port,ssl_context=self.create_ssl_context())
        await site.start()
        await self.print_info() 
        # 保持服务器运行
        await asyncio.Future()

    async def status_handler(self, request):
        '''返回服务器状态信息'''
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "clients_count": len(self.clients),
            "rooms_count": len(self.rooms),
            "rooms": {
                room_id: {
                    "peer_count": len(peers),
                    "peers": list(peers)
                }
                for room_id, peers in self.rooms.items()
            }
        }
        return web.json_response(status)

    async def rooms_handler(self, request):
        '''返回所有房间信息'''
        rooms_info = []
        for room_id, peers in self.rooms.items():
            rooms_info.append({
                "room_id": room_id,
                "peer_count": len(peers),
                "peers": list(peers)
            })
        
        return web.json_response({
            "total_rooms": len(rooms_info),
            "rooms": rooms_info
        })

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
                        logger.error(f"消息解析错误：{msg.data}, 错误：{e}")
                        await self.send_to(ws, {
                            "type": "error",
                            "message": "消息格式错误"
                        })
                    except Exception as e:
                        logger.error(f"处理消息时发生错误：{e}")
                        await self.send_to(ws, {
                            "type": "error",
                            "message": f"处理消息失败：{str(e)}"
                        })
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

    async def handle_message(self, ws, data: dict, peer_id):
        msg_type = data.get("type")

        if msg_type == "register":
            return await self.handle_register(ws, data)
        elif msg_type == "offer":
            await self.handle_offer(data)
        elif msg_type == "answer":
            await self.handle_answer(data)
        elif msg_type == "ice-candidate":
            await self.handle_ice_candidate(data)
        elif msg_type == "call-request":
            await self.handle_call_request(data)
        elif msg_type == "call-accepted":
            await self.handle_call_accepted(data)
        elif msg_type == "call-rejected":
            await self.handle_call_rejected(data)
        elif msg_type == "hangup":
            await self.handle_hangup(data)
        elif msg_type == "get-peer-list":
            await self.handle_get_peer_list(ws, data)
        else:
            print(f"未知消息类型：{msg_type}")
        return peer_id

    async def handle_register(self, ws, data: dict):
        '''
        处理注册消息
        '''
        peer_id = data.get("peerId")
        room_id = data.get("roomId")

        if peer_id in self.clients:
            await self.send_to(ws, {
                "type": "error",
                "message": "Peer ID 已存在"
            })
            return peer_id

        self.clients[peer_id] = {"ws": ws, "roomId": room_id}

        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(peer_id)

        await self.send_to(ws, {
            "type": "welcome",
            "peerId": peer_id,
            "roomId": room_id
        })

        await self.send_peer_list(ws, room_id)

        await self.broadcast_to_room(room_id, peer_id, {
            "type": "peer-joined",
            "peerId": peer_id
        })

        logger.info(f"客户端注册成功：{peer_id}, 房间：{room_id}, 房间人数：{len(self.rooms[room_id])}")
        return peer_id

    async def handle_offer(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        sdp = data.get("sdp")
        await self.forward_to_peer(to_peer, {
            "type": "offer",
            "sdp": sdp,
            "from": from_peer
        })

    async def handle_answer(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        sdp = data.get("sdp")
        await self.forward_to_peer(to_peer, {
            "type": "answer",
            "sdp": sdp,
            "from": from_peer
        })

    async def handle_ice_candidate(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        candidate = data.get("candidate")
        await self.forward_to_peer(to_peer, {
            "type": "ice-candidate",
            "candidate": candidate,
            "from": from_peer
        })

    async def handle_call_request(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        await self.forward_to_peer(to_peer, {
            "type": "call-request",
            "from": from_peer
        })

    async def handle_call_accepted(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        await self.forward_to_peer(to_peer, {
            "type": "call-accepted",
            "from": from_peer
        })

    async def handle_call_rejected(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        await self.forward_to_peer(to_peer, {
            "type": "call-rejected",
            "from": from_peer
        })

    async def handle_hangup(self, data: dict):
        to_peer = data.get("to")
        from_peer = data.get("from")
        await self.forward_to_peer(to_peer, {
            "type": "hangup",
            "from": from_peer
        })

    async def handle_get_peer_list(self, ws, data: dict):
        '''
        获取房间中的用户列表
        '''
        room_id = data.get("roomId")
        peer_id = data.get("peerId")
        
        if not room_id:
            client_info = self.clients.get(peer_id)
            if client_info:
                room_id = client_info.get("roomId")
        
        if room_id and room_id in self.rooms:
            peers = list(self.rooms[room_id])
            await self.send_to(ws, {
                "type": "peer-list",
                "peers": peers
            })

    async def handle_disconnect(self, ws: WebSocketServerProtocol, peer_id: str):
        disconnected_peer_id = peer_id
        disconnected_room_id = self.clients.get(peer_id, {}).get("roomId")

        if disconnected_peer_id in self.clients:
            del self.clients[disconnected_peer_id]

        if disconnected_room_id and disconnected_room_id in self.rooms:
            room = self.rooms[disconnected_room_id]
            room.discard(disconnected_peer_id)

            if len(room) == 0:
                del self.rooms[disconnected_room_id]
            else:
                await self.broadcast_to_room(disconnected_room_id, disconnected_peer_id, {
                    "type": "peer-left",
                    "peerId": disconnected_peer_id
                })
                logger.info(f"用户 {disconnected_peer_id} 离开房间 {disconnected_room_id}, 剩余人数：{len(room)}")

        logger.info(f"客户端断开连接：{disconnected_peer_id}")

    async def forward_to_peer(self, peer_id: str, message: dict):
        if peer_id in self.clients:
            client = self.clients[peer_id]
            await self.send_to(client["ws"], message)

    async def send_to(self, ws, message: dict):
        try:
            await ws.send_str(json.dumps(message))
        except Exception as e:
            print(f"发送消息错误: {e}")

    async def send_peer_list(self, ws, room_id: str):
        if room_id in self.rooms:
            peers = list(self.rooms[room_id])
            await self.send_to(ws, {
                "type": "peer-list",
                "peers": peers
            })

    async def broadcast_to_room(self, room_id: str, exclude_peer_id: str, message: dict):
        if room_id in self.rooms:
            for peer_id in self.rooms[room_id]:
                if peer_id != exclude_peer_id:
                    await self.forward_to_peer(peer_id, message)


if __name__ == "__main__":
    from model.apphelp import get_config
    config = get_config()
    port = config.get("port")  
    server = SignalingServer(port)   
    if config.get("ssl"):
        server.use_ssl(config.get("cert"), config.get("key"))
    asyncio.run(server.start())
