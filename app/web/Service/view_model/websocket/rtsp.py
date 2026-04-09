from co6co.task.eventDispatcher import EventDispatcherProcess
from sanic import Sanic,Request
from sanic.server.websockets.impl import WebsocketImplProtocol as WebSocketCommonProtocol
from services.rtspService import RtspServiceClient
from services.ff_service import ffService
from co6co.utils import log
from co6co.task.thread import TaskManage

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
        rtsp_url = request.args.get('url') 
        print(rtsp_url)
        if not rtsp_url:
            await ws.close(reason="需要RTSP URL参数")
            return
        
        # 创建唯一标识符
        import uuid
        key = str(uuid.uuid4())
        _service = ffService() 
        taskMgr=TaskManage(threadName="rtspStreamTask")
        async def startStream(key, rtsp_url): 
            async for data in _service.read_rtsp_stream(rtsp_url,key):  
                await ws.send(data)
        taskMgr.runTask(startStream,lambda x:print(f"[INFO] 流任务: {key} 结束，执行结果：{x.result}"),key, rtsp_url)
        while True:
            message = await ws.recv() 
            if message == "close":
                print(f"[INFO] 收到关闭指令")
                await ws.close(reason="关闭指令") 
                _service.stop(key)
                taskMgr.stop()
                break 

    @staticmethod
    async def websocket_stream0(request:Request, ws: WebSocketCommonProtocol):
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
            log.warn(f"[INFO] 111111111111111111: {ws_key}")
            while True:
                log.warn(f"循环中recv 前  2222222222222222: {ws_key}")
                message = await ws.recv()
                log.warn(f"循环中recv 后 222222222222222222.1: {ws_key}")
                if message == "close":
                    print(f"[INFO] 收到关闭指令")
                    await ws.close(reason="关闭指令")
                    log.warn(f"[INFO]收到 close 222222222222222222.3: {ws_key}") 
                    break
                # 可以处理其他控制消息
                # 例如: 暂停、恢复、改变参数等
                log.warn(f"[INFO] {ws_key}收到其他指令: {message}")
            print(f"[INFO] 退出循环: {ws_key}")
        except Exception as e:
            print(f"[INFO] ws连接断开: {e}")
            log.warn(f"异常 99999999999999999999: {ws_key}") 
        finally:
            client.send_stop(ws_key)
            log.warn(f"最终 1010101010101010101010: {ws_key}")
         


    