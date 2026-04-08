from co6co.task.eventDispatcher import EventDispatcherProcess
from sanic import Sanic,Request
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
from services.rtspService import RtspServiceClient
from services.ff_service import ffService


class client_vod: 
    routePath="/stream" 
    def get_client(request:Request ):
        event_process: EventDispatcherProcess =request.app.ctx.event_process 
        return event_process 
    @staticmethod
    async def websocket_stream2(request:Request, ws: WebSocketCommonProtocol):
        rtsp_url = request.args.get('url') 
        print(rtsp_url)
        if not rtsp_url:
            await ws.close(reason="需要RTSP URL参数")
            return
        
        # 创建唯一标识符
        import uuid
        ws_key = str(uuid.uuid4())
        ffservice=ffService()
        async for  data in ffservice.read_rtsp_stream(rtsp_url,ws_key):
            await ws.send(data)
        while True:
                message = await ws.recv()
                if message == "close":
                    print(f"[INFO] 收到关闭指令")
                    client.send_stop(ws_key)
                    break
                # 可以处理其他控制消息
                # 例如: 暂停、恢复、改变参数等 
        

    @staticmethod
    async def websocket_stream(request:Request, ws: WebSocketCommonProtocol):
        """WebSocket流端点""" 
        rtsp_url = request.args.get('url') 
        print(rtsp_url)
        if not rtsp_url:
            await ws.close(reason="需要RTSP URL参数")
            return
        
        # 创建唯一标识符
        import uuid
        ws_key = str(uuid.uuid4())
        # 启动流任务 
        msg_process = client_vod.get_client(request) 
        client = RtspServiceClient( ) 
        try: 
            if not msg_process.dispatcher.exist(client.key):
                msg_process.append_handler(client) 
            else:
                client=msg_process.dispatcher.get_handler(client.key) 
            client.send_start(ws,ws_key,rtsp_url) 
            # 保持连接活跃
            while True:
                message = await ws.recv()
                if message == "close":
                    print(f"[INFO] 收到关闭指令")
                    client.send_stop(ws_key)
                    break
                # 可以处理其他控制消息
                # 例如: 暂停、恢复、改变参数等 
        except Exception as e:
            print(f"[INFO] ws连接断开: {e}")
            raise e
         


    