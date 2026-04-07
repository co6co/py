from co6co.task.eventDispatcher import EventDispatcherProcess
from sanic import Sanic,Request
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
from services.rtspService import RtspServiceClient


class client_vod: 
    routePath="/stream" 
    def get_client(self,request:Request ):
        event_process: EventDispatcherProcess =request.app.ctx.event_process 
        return event_process 
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
        msg_process = client_vod.get_client(request,ws)
        client = RtspServiceClient( )  
        try: 
            msg_process.append_handler(ws_key, client)
            client.send_start(ws,ws_key,rtsp_url) 
            # 保持连接活跃
            while True:
                message = await ws.recv()
                if message == "close":
                    print(f"[INFO] 收到关闭指令")
                    stream_task.cancel()
                    break
                # 可以处理其他控制消息
                # 例如: 暂停、恢复、改变参数等
                
        except Exception as e:
            print(f"[INFO] 连接断开: {e}")
        finally:
            # 取消任务并清理
            #task = app.ctx.background_tasks.pop(ws_key, None)
            msg_process.remove_handler(ws_key,client)
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            
            rtsService.stop(ws_key)
            print(f"[INFO] 连接清理完成: {ws_key}")


    