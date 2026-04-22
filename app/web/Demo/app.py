# server.py
import asyncio
import json
import subprocess
from sanic import Sanic
from sanic.response import json as sanic_json
#from sanic.websocket import WebSocketCommonProtocol
from websockets.legacy.protocol import WebSocketCommonProtocol
from model.apphelp import get_config

app = Sanic("RTSPtoWebSocketProxy")

# 全局字典，管理 WebSocket 连接和对应的 FFmpeg 进程
ws_connections = {}

async def stream_video_to_ws(ws: WebSocketCommonProtocol, rtsp_url: str):
    """启动 FFmpeg 进程，并将输出流通过 WebSocket 发送"""
    # 关键：使用正确的参数生成碎片化的 MP4 流，便于前端处理
    ffmpeg_cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',  # 使用 TCP 传输，提高稳定性
        '-i', rtsp_url,            # RTSP 源地址
        '-f', 'mp4',               # 输出格式为 MP4
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof', # 生成碎片化 MP4
        '-c:v', 'libx264',         # 视频编码
        '-preset', 'ultrafast',    # 最快编码速度，降低延迟
        '-tune', 'zerolatency',    # 零延迟优化
        '-b:v', '500k',            # 视频比特率，可根据需要调整
        '-frag_duration', '100000', # 每个碎片的微秒时长
        '-reset_timestamps', '1',
        'pipe:1'                   # 输出到标准输出 (stdout)
    ]

    process = None
    try:
        # 启动 FFmpeg 进程
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 忽略错误输出，或重定向到文件以便调试
        )

        # 持续读取 FFmpeg 的标准输出并发送
        while True:
            chunk = await process.stdout.read(65536)  # 每次读取 64KB
            if not chunk:
                print(f"[INFO] FFmpeg 进程输出结束 for {rtsp_url}")
                break
            try:
                await ws.send(chunk)
            except Exception as e:
                print(f"[ERROR] WebSocket 发送失败: {e}")
                break
    except asyncio.CancelledError:
        print(f"[INFO] 流任务被取消 for {rtsp_url}")
    except Exception as e:
        print(f"[ERROR] 流处理异常: {e}")
    finally:
        # 清理：终止 FFmpeg 进程
        if process and process.returncode is None:
            try:
                process.terminate()
                await process.wait()
            except Exception as e:
                print(f"[WARN] 终止 FFmpeg 进程时出错: {e}")
        print(f"[INFO] 清理完成 for {rtsp_url}")

@app.websocket('/ws/stream')
async def feed(request, ws):
    """WebSocket 路由，处理客户端连接"""
    rtsp_url = request.args.get('url')
    if not rtsp_url:
        await ws.close(reason="未提供 RTSP URL 参数")
        return

    print(f"[INFO] 新的 WebSocket 连接，RTSP URL: {rtsp_url}")

    # 为当前连接创建并存储流任务
    stream_task = asyncio.create_task(stream_video_to_ws(ws, rtsp_url))
    ws_connections[ws] = stream_task

    try:
        # 保持连接，直到客户端断开
        while True:
            msg = await ws.recv()
            # 这里可以处理客户端发来的控制消息，例如：暂停、恢复、改变画质等
            # 例如: data = json.loads(msg)
            # 但本示例中，我们只维持连接，不做复杂控制。
            print(f"[DEBUG] 收到客户端消息: {msg}")
    except Exception as e:
        # 当 recv() 抛出异常（通常是连接断开）时跳出循环
        print(f"[INFO] WebSocket 连接断开: {e}")
    finally:
        # 连接断开，取消对应的流任务并进行清理
        print(f"[INFO] 开始清理连接资源 for {rtsp_url}")
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
        ws_connections.pop(ws, None)
        print(f"[INFO] 连接资源清理完成 for {rtsp_url}")

if __name__ == '__main__':
    config=get_config()
    port=config.get("port")
    app.run(host="0.0.0.0", port=port, debug=False, access_log=False)
