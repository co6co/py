# websocket_server.py
import asyncio
import json
import time
import uuid
import signal
import sys
from datetime import datetime
from typing import Dict, Set, Optional

from sanic import Sanic, Request, response
from sanic.response import html, json as sanic_json

from co6co_sanic_ext.cors import CORS,cross_origin
#from websockets.legacy.protocol import WebSocketCommonProtocol
#from sanic.ext.websocket import WebSocketServer as WebSocketCommonProtocol
from model.utils import get_client_ip, check_connection_alive

from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol

from model.apphelp import read_file_content,get_file_path 

# 创建 Sanic 应用
app = Sanic("WebSocketDemo")
CORS(app, resources={r"/*": {"origins": "*"}})

# 存储连接的客户端
connected_clients: Set[WebSocketCommonProtocol] = set()
client_info: Dict[WebSocketCommonProtocol, dict] = {}

# 房间管理
rooms: Dict[str, Set[WebSocketCommonProtocol]] = {}

async def broadcast(message: dict, exclude: Optional[WebSocketCommonProtocol] = None):
    """广播消息给所有客户端"""
    message_str = json.dumps(message)
    disconnected = []
    
    for client in connected_clients:
        if client != exclude :#:and client.open:
            try:
                await client.send(message_str)
            except:
                disconnected.append(client)
    
    # 清理断开连接的客户端
    for client in disconnected:
        await cleanup_client(client)

async def broadcast_to_room(room_id: str, message: dict, exclude: Optional[WebSocketCommonProtocol] = None):
    """广播消息给指定房间的客户端"""
    if room_id not in rooms:
        return
    
    message_str = json.dumps(message)
    disconnected = []
    
    for client in rooms[room_id]:
        
        if client != exclude :#and client.open:
            try:
                await client.send(message_str)
            except:
                disconnected.append(client)
    
    # 清理断开连接的客户端
    for client in disconnected:
        await cleanup_client(client, room_id)

async def send_to_client(client: WebSocketCommonProtocol, message: dict):
    """发送消息给指定客户端"""
    print(f"[INFO] 发送消息给客户端: {client}, {type(client)}")
    
    #if client.open:
    try:
        await client.send(json.dumps(message))
    except:
        await cleanup_client(client)

async def cleanup_client(client: WebSocketCommonProtocol, room_id: str = None):
    """清理客户端连接"""
    if client in connected_clients:
        connected_clients.remove(client)
    
    if client in client_info:
        client_data = client_info.pop(client, {})
        left_room = client_data.get('room_id')
        
        # 从房间中移除
        if left_room and left_room in rooms and client in rooms[left_room]:
            rooms[left_room].remove(client)
            
            # 如果房间为空，删除房间
            if not rooms[left_room]:
                rooms.pop(left_room, None)
            
            # 广播用户离开
            if left_room:
                await broadcast_to_room(left_room, {
                    "type": "user_left",
                    "client_id": client_data.get('client_id'),
                    "username": client_data.get('username'),
                    "timestamp": datetime.now().isoformat(),
                    "room_id": left_room
                }, exclude=client)




async def handle_system_message(request, client: WebSocketCommonProtocol, data: dict):
    """处理系统消息"""
    message_type = data.get('type')
    client_data = client_info.get(client, {})
    ip = get_client_ip(request)
    if message_type == 'ping':
        # 心跳响应
        await send_to_client(client, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == 'join_room':
        # 加入房间
        room_id = data.get('room_id', 'default')
        username = data.get('username', f'用户{str(uuid.uuid4())[:8]}')
        
        # 离开之前的房间
        old_room = client_data.get('room_id')
        if old_room and old_room in rooms and client in rooms[old_room]:
            rooms[old_room].remove(client)
            await broadcast_to_room(old_room, {
                "type": "user_left",
                "client_id": client_data.get('client_id'),
                "username": client_data.get('username'),
                "timestamp": datetime.now().isoformat(),
                "room_id": old_room
            }, exclude=client)
        
        # 加入新房间
        if room_id not in rooms:
            rooms[room_id] = set()
        rooms[room_id].add(client)
        
        # 更新客户端信息
        client_info[client] = {
            "client_id": client_data.get('client_id', str(uuid.uuid4())),
            "username": username,
            "room_id": room_id,
            "joined_at": datetime.now().isoformat(),
            "ip": ip
        }
        
        # 发送加入成功消息
        await send_to_client(client, {
            "type": "room_joined",
            "room_id": room_id,
            "username": username,
            "client_id": client_info[client]["client_id"],
            "timestamp": datetime.now().isoformat(),
            "room_users": len(rooms[room_id])
        })
        
        # 广播新用户加入
        await broadcast_to_room(room_id, {
            "type": "user_joined",
            "client_id": client_info[client]["client_id"],
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "room_id": room_id
        }, exclude=client)
    
    elif message_type == 'leave_room':
        # 离开房间
        room_id = client_data.get('room_id')
        if room_id and room_id in rooms and client in rooms[room_id]:
            await cleanup_client(client, room_id)
    
    elif message_type == 'get_room_info':
        # 获取房间信息
        room_id = data.get('room_id', client_data.get('room_id', 'default'))
        users = []
        
        if room_id in rooms:
            for room_client in rooms[room_id]:
                if room_client in client_info:
                    users.append(client_info[room_client])
        
        await send_to_client(client, {
            "type": "room_info",
            "room_id": room_id,
            "users": users,
            "user_count": len(users),
            "timestamp": datetime.now().isoformat()
        })

@app.websocket('/ws')
async def websocket_handler(request: Request, ws: WebSocketCommonProtocol):
    """WebSocket 主处理器"""
    # 添加到连接列表
    connected_clients.add(ws)
    client_id = str(uuid.uuid4())
    ip = get_client_ip(request)
    # 初始化客户端信息
    client_info[ws] = {
        "client_id": client_id,
        "username": f"用户{client_id[:8]}",
        "room_id": None,
        "joined_at": datetime.now().isoformat(),
        "ip":ip
    }
    
    # 发送欢迎消息
    await send_to_client(ws, {
        "type": "welcome",
        "client_id": client_id,
        "message": "已连接到WebSocket服务器",
        "timestamp": datetime.now().isoformat(),
        "server_info": {
            "name": "Sanic WebSocket Server",
            "version": "1.0.0",
            "connected_clients": len(connected_clients)
        }
    })
    
    # 广播新连接
    await broadcast({
        "type": "client_connected",
        "client_id": client_id,
        "timestamp": datetime.now().isoformat(),
        "total_clients": len(connected_clients)
    }, exclude=ws)
    
    print(f"[INFO] 客户端连接: {client_id} ({request.ip}) - 总连接数: {len(connected_clients)}")
    
    try:
        # 主消息循环
        while True:
            message = await ws.recv()
            
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type in ['ping', 'join_room', 'leave_room', 'get_room_info']:
                    # 系统消息
                    await handle_system_message(request, ws, data)
                
                elif message_type == 'chat_message':
                    # 聊天消息
                    client_data = client_info.get(ws, {})
                    room_id = client_data.get('room_id')
                    
                    if room_id:
                        await broadcast_to_room(room_id, {
                            "type": "chat_message",
                            "client_id": client_data.get('client_id'),
                            "username": client_data.get('username'),
                            "message": data.get('message', ''),
                            "timestamp": datetime.now().isoformat(),
                            "room_id": room_id
                        }, exclude=ws)
                    else:
                        await send_to_client(ws, {
                            "type": "error",
                            "message": "请先加入一个房间",
                            "timestamp": datetime.now().isoformat()
                        })
                
                elif message_type == 'broadcast':
                    # 广播消息
                    client_data = client_info.get(ws, {})
                    await broadcast({
                        "type": "broadcast",
                        "client_id": client_data.get('client_id'),
                        "username": client_data.get('username'),
                        "message": data.get('message', ''),
                        "timestamp": datetime.now().isoformat()
                    }, exclude=ws)
                
                elif message_type == 'private_message':
                    # 私聊消息
                    target_client_id = data.get('target_client_id')
                    target_client = None
                    
                    # 查找目标客户端
                    for client, info in client_info.items():
                        if info.get('client_id') == target_client_id:
                            target_client = client
                            break
                    
                    if target_client :#and target_client.open:
                        client_data = client_info.get(ws, {})
                        await send_to_client(target_client, {
                            "type": "private_message",
                            "from_client_id": client_data.get('client_id'),
                            "from_username": client_data.get('username'),
                            "message": data.get('message', ''),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # 发送回执
                        await send_to_client(ws, {
                            "type": "private_message_sent",
                            "to_client_id": target_client_id,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        await send_to_client(ws, {
                            "type": "error",
                            "message": f"目标客户端 {target_client_id} 不存在或已断开连接",
                            "timestamp": datetime.now().isoformat()
                        })
                
                else:
                    # 回显其他消息
                    await send_to_client(ws, {
                        "type": "echo",
                        "original_message": data,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except json.JSONDecodeError:
                # 非JSON消息，直接回显
                await send_to_client(ws, {
                    "type": "echo",
                    "original_message": message,
                    "timestamp": datetime.now().isoformat()
                })
            
    except Exception as e:
        # 连接断开
        print(f"[INFO] 客户端断开: {client_id} - 原因: {type(e).__name__}: {str(e)}")
    
    finally:
        # 清理连接
        await cleanup_client(ws)
        print(f"[INFO] 客户端清理完成: {client_id} - 剩余连接: {len(connected_clients)}")

@app.websocket('/ws/echo')
async def echo_handler(request: Request, ws: WebSocketCommonProtocol):
    """简单的回显 WebSocket 端点"""
    print(f"[INFO] Echo WebSocket 连接: {request.ip}")
    
    try:
        while True:
            data = await ws.recv()
            await ws.send(f"服务器收到: {data} - 时间: {datetime.now().isoformat()}")
    except Exception as e:
        print(f"[INFO] Echo 连接断开: {e}")

@app.websocket('/ws/chat')
async def chat_handler(request: Request, ws: WebSocketCommonProtocol):
    """简单的聊天室 WebSocket 端点"""
    connected_clients.add(ws)
    client_id = str(uuid.uuid4())
    
    print(f"[INFO] Chat WebSocket 连接: {client_id} ({request.ip})")
    
    # 发送欢迎消息
    await ws.send(json.dumps({
        "type": "welcome",
        "client_id": client_id,
        "message": "欢迎来到聊天室!",
        "timestamp": datetime.now().isoformat()
    }))
    
    # 广播新用户加入
    for client in connected_clients:
        isOpen=await check_connection_alive(client)
        if client != ws and isOpen:
            try:
                await client.send(json.dumps({
                    "type": "user_joined",
                    "client_id": client_id,
                    "timestamp": datetime.now().isoformat()
                }))
            except:
                pass
    
    try:
        while True:
            data = await ws.recv()
            message_data = json.loads(data) if data else {}
            
            # 广播消息
            for client in connected_clients:
                isOpen=await check_connection_alive(client)
        
                if isOpen:
                    try:
                        await client.send(json.dumps({
                            "type": "message",
                            "client_id": client_id,
                            "message": message_data.get('message', data),
                            "timestamp": datetime.now().isoformat()
                        }))
                    except:
                        pass
    
    except Exception as e:
        print(f"[INFO] Chat 连接断开: {client_id} - {e}")
    
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)
        
        # 广播用户离开
        for client in connected_clients:
            isOpen=await check_connection_alive(client)
            if isOpen:
                try:
                    await client.send(json.dumps({
                        "type": "user_left",
                        "client_id": client_id,
                        "timestamp": datetime.now().isoformat()
                    }))
                except:
                    pass

@app.route('/api/status')
async def get_status(request: Request):
    """获取服务器状态"""
    return sanic_json({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "server": "Sanic WebSocket Server",
        "connected_clients": len(connected_clients),
        "rooms": {room_id: len(clients) for room_id, clients in rooms.items()},
        "clients": list(client_info.values())
    })

@app.route('/api/send_to_all', methods=['POST'])
async def send_to_all(request: Request):
    """向所有客户端发送消息（管理员接口）"""
    data = request.json
    if not data or 'message' not in data:
        return sanic_json({"error": "缺少消息内容"}, status=400)
    
    await broadcast({
        "type": "admin_message",
        "message": data['message'],
        "timestamp": datetime.now().isoformat()
    })
    
    return sanic_json({"success": True, "sent_to": len(connected_clients)})

@app.route('/')
async def index(request: Request):
    """主页面"""
    index_path=get_file_path('demoIndex.html')
    html_content =   read_file_content(index_path)
    return html(html_content)

# 优雅关闭
def signal_handler(sig, frame):
    print("\n[INFO] 收到关闭信号，清理资源...")
    asyncio.create_task(cleanup_all())
    sys.exit(0)

async def cleanup_all():
    """清理所有连接"""
    for client in list(connected_clients):
        await cleanup_client(client)
    print(f"[INFO] 已清理所有连接")

# 静态文件服务
app.static('/static', './static')

if __name__ == '__main__':
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建静态目录
    import os
    os.makedirs('static/js', exist_ok=True)
    
    # 运行服务器
    app.run(
        host="0.0.0.0",
        port=8800,
        debug=True,
        access_log=True,
        auto_reload=True
    )