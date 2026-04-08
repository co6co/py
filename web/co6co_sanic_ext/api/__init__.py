
from sanic import Sanic, Blueprint, Request
from ..view_model import BaseView
from typing import Callable
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
def add_routes(blue: Blueprint, *views: BaseView):
    """
    增加路由
    """
    for v in views:
        blue.add_route(v.as_view(), v.routePath, name=v.__name__)
 
def add_websocket_route(blue: Blueprint,hander:Callable[[Request, WebSocketCommonProtocol], None],routePath:str = None,**kwargs):
    blue.add_websocket_route(hander, routePath,**kwargs)

'''
class ChatWebSocket:
    routePath = "/chat" 
    async def ws_handler(self, request, ws):
        """WebSocket 处理"""
        await ws.send("欢迎来到聊天室！")
        
        async for msg in ws:
            if msg == "exit":
                await ws.send("再见！")
                break
            await ws.send(f"收到消息: {msg}")
    
    async def http_get(self, request: Request):
        """HTTP GET 方法，返回 WebSocket 连接页面"""
        return text("""
        <html>
            <script>
                const ws = new WebSocket('ws://' + window.location.host + '/index/chat');
                ws.onmessage = (e) => console.log(e.data);
                ws.onopen = () => ws.send('Hello');
            </script>
        </html>
        """)

# 使用
chat_view = ChatWebSocket()
# 添加 WebSocket 路由
blue.add_websocket_route(chat_view.ws_handler, chat_view.routePath)
# 添加 HTTP 路由
blue.add_route(chat_view.http_get, chat_view.routePath, methods=["GET"])
'''