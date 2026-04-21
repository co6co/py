# simple_websocket_server.py
import asyncio
import json
from aiohttp import web
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 存储所有连接的客户端
connected_clients = set()


async def websocket_handler(request):
    """WebSocket处理器"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_ip = request.remote
    logger.info(f"新的WebSocket连接: {client_ip}")

    # 添加到连接列表
    connected_clients.add(ws)

    try:
        # 发送欢迎消息
        await ws.send_str(json.dumps({
            "type": "welcome",
            "message": f"欢迎连接到WebSocket服务器",
            "clients": len(connected_clients)
        }))

        # 处理消息
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    logger.info(f"收到来自 {client_ip} 的消息: {data}")

                    # 回应
                    response = {
                        "type": "echo",
                        "message": f"收到你的消息: {data.get('message', '')}",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await ws.send_str(json.dumps(response))

                except json.JSONDecodeError:
                    logger.error(f"JSON解析失败: {msg.data}")
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": "无效的JSON格式"
                    }))

            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket连接错误: {ws.exception()}")

    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")

    finally:
        # 从连接列表移除
        connected_clients.discard(ws)
        logger.info(f"WebSocket连接关闭: {client_ip}")
        logger.info(f"当前连接数: {len(connected_clients)}")

    return ws


async def index_handler(request):
    """主页，提供测试页面"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket测试</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .panel { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin-right: 10px; }
            .status { margin: 10px 0; padding: 10px; border-radius: 4px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            .messages { height: 300px; overflow-y: auto; background: white; padding: 10px; border: 1px solid #ddd; }
            .message { margin: 5px 0; padding: 5px; background: #e9ecef; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WebSocket连接测试</h1>
            
            <div class="panel">
                <h2>连接信息</h2>
                <div id="status" class="status disconnected">未连接</div>
                <div id="info"></div>
            </div>
            
            <div class="controls">
                <button onclick="connectWebSocket()">连接WebSocket</button>
                <button onclick="sendMessage()" id="sendBtn" disabled>发送消息</button>
                <button onclick="disconnectWebSocket()" id="disconnectBtn" disabled>断开连接</button>
                <button onclick="clearMessages()">清空消息</button>
            </div>
            
            <div class="panel">
                <h2>消息面板</h2>
                <input type="text" id="messageInput" placeholder="输入消息..." style="width: 300px; padding: 8px;">
                <div class="messages" id="messages"></div>
            </div>
            
            <div class="panel">
                <h2>测试消息</h2>
                <button onclick="sendPing()">发送Ping</button>
                <button onclick="sendTestData()">发送测试数据</button>
            </div>
        </div>

        <script>
            let ws = null;
            const messagesDiv = document.getElementById('messages');
            const statusDiv = document.getElementById('status');
            const infoDiv = document.getElementById('info');
            
            function logMessage(message, isReceived = false) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message';
                const prefix = isReceived ? '接收: ' : '发送: ';
                msgDiv.textContent = prefix + message;
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function updateStatus(status, isConnected = false) {
                statusDiv.textContent = status;
                statusDiv.className = isConnected ? 'status connected' : 'status disconnected';
                document.getElementById('sendBtn').disabled = !isConnected;
                document.getElementById('disconnectBtn').disabled = !isConnected;
            }
            
            function updateInfo(info) {
                infoDiv.textContent = info;
            }
            
            function connectWebSocket() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    alert('已经连接了！');
                    return;
                }
                
                updateStatus('连接中...', false);
                
                // 获取当前页面的主机和端口
                const host = window.location.host;
                ws = new WebSocket(`ws://${host}/ws`);
                
                ws.onopen = function(event) {
                    updateStatus('已连接到WebSocket服务器', true);
                    updateInfo(`连接时间: ${new Date().toLocaleTimeString()}`);
                    logMessage('WebSocket连接已打开', false);
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        logMessage(JSON.stringify(data, null, 2), true);
                    } catch (e) {
                        logMessage(event.data, true);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket错误:', error);
                    updateStatus('连接错误', false);
                };
                
                ws.onclose = function(event) {
                    updateStatus('连接已关闭', false);
                    logMessage(`连接关闭，代码: ${event.code}, 原因: ${event.reason || '无'}`, false);
                    ws = null;
                };
            }
            
            function sendMessage() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('WebSocket未连接！');
                    return;
                }
                
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) {
                    alert('请输入消息！');
                    return;
                }
                
                const data = {
                    type: 'message',
                    message: message,
                    timestamp: Date.now()
                };
                
                ws.send(JSON.stringify(data));
                logMessage(JSON.stringify(data));
                input.value = '';
            }
            
            function disconnectWebSocket() {
                if (ws) {
                    ws.close(1000, '用户主动断开');
                    ws = null;
                }
            }
            
            function clearMessages() {
                messagesDiv.innerHTML = '';
            }
            
            function sendPing() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('WebSocket未连接！');
                    return;
                }
                
                const data = {
                    type: 'ping',
                    timestamp: Date.now()
                };
                
                ws.send(JSON.stringify(data));
                logMessage(JSON.stringify(data));
            }
            
            function sendTestData() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('WebSocket未连接！');
                    return;
                }
                
                const testData = {
                    type: 'test',
                    data: {
                        number: Math.random(),
                        text: '测试消息',
                        array: [1, 2, 3],
                        object: { key: 'value' }
                    },
                    timestamp: Date.now()
                };
                
                ws.send(JSON.stringify(testData));
                logMessage(JSON.stringify(testData));
            }
            
            // 允许按Enter键发送消息
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // 页面加载时显示当前时间
            updateInfo(`页面加载时间: ${new Date().toLocaleTimeString()}`);
        </script>
    </body>
    </html>
    '''
    return web.Response(text=html, content_type="text/html")


async def stats_handler(request):
    """统计信息"""
    return web.json_response({
        "clients": len(connected_clients),
        "timestamp": asyncio.get_event_loop().time(),
        "status": "running"
    })


async def broadcast_handler(request):
    """广播消息给所有客户端（测试用）"""
    data = await request.json()
    message = data.get("message", "广播消息")

    for client in connected_clients:
        try:
            await client.send_str(json.dumps({
                "type": "broadcast",
                "message": message,
                "timestamp": asyncio.get_event_loop().time()
            }))
        except Exception as e:
            logger.error(f"广播消息失败: {e}")

    return web.json_response({
        "status": "success",
        "clients": len(connected_clients)
    })


async def cleanup_clients():
    """定期清理断开的连接"""
    while True:
        await asyncio.sleep(60)  # 每分钟检查一次
        initial_count = len(connected_clients)
        connected_clients = {client for client in connected_clients if not client.closed}
        removed = initial_count - len(connected_clients)
        if removed > 0:
            logger.info(f"清理了 {removed} 个断开的连接")


async def startup(app):
    """启动时初始化"""
    logger.info("WebSocket服务器启动中...")
    # 启动清理任务
    asyncio.create_task(cleanup_clients())


async def shutdown(app):
    """关闭时清理"""
    logger.info("WebSocket服务器关闭中...")
    # 关闭所有客户端连接
    for client in connected_clients:
        await client.close()
    connected_clients.clear()


def create_app():
    """创建应用"""
    app = web.Application()

    # 添加路由
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/stats', stats_handler)
    app.router.add_post('/broadcast', broadcast_handler)

    # 启动和关闭钩子
    app.on_startup.append(startup)
    app.on_shutdown.append(shutdown)

    return app


if __name__ == '__main__':
    app = create_app()

    # 配置CORS（如果需要）
    from aiohttp_cors import setup, ResourceOptions

    cors = setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # 为所有路由启用CORS
    for route in list(app.router.routes()):
        cors.add(route)

    logger.info("启动WebSocket服务器: http://0.0.0.0:8901")
    logger.info("测试页面: http://localhost:8901")
    logger.info("WebSocket端点: ws://localhost:8901/ws")
    logger.info("状态接口: http://localhost:8901/stats")

    web.run_app(app, host='0.0.0.0', port=8902, access_log=None)
