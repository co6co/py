import asyncio,json
from sanic import Request
from sanic.server.websockets.impl import WebsocketImplProtocol  
def get_client_ip(request: Request) -> str:
    """获取客户端IP的正确方法"""
    # 尝试多种方式获取IP
    ip = None
    
    # 1. 从X-Forwarded-For头部获取（如果有反向代理）
    if request.headers.get('X-Forwarded-For'):
        ip = request.headers.get('X-Forwarded-For').split(',')[0].strip()
    
    # 2. 从X-Real-IP头部获取
    elif request.headers.get('X-Real-IP'):
        ip = request.headers.get('X-Real-IP')
    
    # 3. 从request对象获取
    elif hasattr(request, 'ip') and request.ip:
        ip = request.ip
    elif hasattr(request, 'remote_addr') and request.remote_addr:
        ip = request.remote_addr
    
    return ip or 'unknown'

async def check_connection_alive(ws: WebsocketImplProtocol) -> bool:
    """检查WebSocket连接是否活跃"""
    try:
        # 方法1: 尝试发送心跳
        await asyncio.wait_for(
            ws.send(json.dumps({"type": "ping"})),
            timeout=1.0
        )
        return True
    except (asyncio.TimeoutError, Exception):
        return False