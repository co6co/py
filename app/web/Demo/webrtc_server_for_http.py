import asyncio
import json
import os
from aiohttp import web
from websockets.server import WebSocketServerProtocol 

class SignalingServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients = {}  # peerId -> {ws, roomId}
        self.rooms = {}    # roomId -> Set of peerIds
        self.websocket = None

    async def start(self):
        app = web.Application()
        app.add_routes([
            web.get('/ws', self.websocket_handler),
            web.static('/', os.path.join(os.path.dirname(__file__), 'pages')),
        ])
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        print(f"信令服务器运行在 http://localhost:{self.port}")
        print(f"WebSocket 端点: ws://localhost:{self.port}/ws")
        
        # 保持服务器运行
        await asyncio.Future()

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print("新的客户端连接")
        peer_id = None

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        peer_id = await self.handle_message(ws, data, peer_id)
                    except json.JSONDecodeError:
                        print(f"消息解析错误: {msg.data}")
                elif msg.type == web.WSMsgType.ERROR:
                    print(f"WebSocket 错误: {ws.exception()}")
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
        else:
            print(f"未知消息类型: {msg_type}")
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

        print(f"客户端注册: {peer_id}, 房间: {room_id}")
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

        print(f"客户端断开: {disconnected_peer_id}")

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
    asyncio.run(server.start())