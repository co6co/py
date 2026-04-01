# minimal_server.py
from aiohttp import web
import aiohttp
import asyncio
import json
from model.apphelp import read_file_content,get_file_path

class WebSocketServer:
    def __init__(self):
        self.app = web.Application()
        self.clients = set()
        self.setup_routes()
    
    def setup_routes(self):
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_static('/page', './pages')
    
    async def handle_index(self, request):
        index_path=get_file_path('miniIndex.html')
        html_content =   read_file_content(index_path)
        return web.Response(text=html_content, content_type='text/html')
    
    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.clients.add(ws)
        print(f"[INFO] 新WebSocket连接，当前连接数: {len(self.clients)}")
        
        await ws.send_json({
            "type": "welcome",
            "message": "连接成功",
            "clients": len(self.clients)
        })
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"收到消息: {data}")
                    
                    # 回显消息
                    await ws.send_json({
                        "type": "echo",
                        "original": data,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    
                elif msg.type == web.WSMsgType.ERROR:
                    print(f"WebSocket错误: {ws.exception()}")
                    
        except Exception as e:
            print(f"连接异常: {e}")
        finally:
            self.clients.remove(ws)
            print(f"[INFO] WebSocket断开，剩余连接数: {len(self.clients)}")
        
        return ws

if __name__ == '__main__':
    server = WebSocketServer()
    web.run_app(server.app, host='0.0.0.0', port=8800)