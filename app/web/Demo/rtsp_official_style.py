import asyncio
import json
import logging
import os
import sys
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
from model.apphelp import get_config  

async def run(pc, player, recorder, duration=10):
    """
    运行录制任务
    """
    # 添加轨道
    if player.video:
        pc.addTrack(player.video)
        recorder.addTrack(player.video)
    
    if player.audio:
        pc.addTrack(player.audio)
        recorder.addTrack(player.audio)
    
    # 创建SDP
    await pc.setLocalDescription(await pc.createOffer())
    
    # 等待录制
    for i in range(duration):
        await asyncio.sleep(1)
        print(f"已录制: {i+1}秒")
    
    # 停止录制
    await recorder.stop()

async def official_style_record():
    """官方示例风格的录制"""
    config=get_config()
    url=config.get("rtsp_url")
    rtsp_url = url
    output_file = "official_output.mp4"
    duration = 10
    
    # 创建对等连接
    pc = RTCPeerConnection()
    
    # 创建媒体播放器
    player = MediaPlayer(
        rtsp_url,
        format="rtsp",
        options={"rtsp_transport": "tcp"}
    )
    
    # 等待播放器初始化
    await asyncio.sleep(2)
    
    if not player.video:
        print("没有视频轨道")
        await pc.close()
        return
    
    # 创建录制器
    recorder = MediaRecorder(output_file)
    
    try:
        # 启动录制器
        await recorder.start()
        
        # 运行录制任务
        await run(pc, player, recorder, duration)
        
        # 检查结果
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ 成功！文件大小: {size}字节")
        else:
            print("❌ 文件未创建")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理
        await pc.close()

# 运行
asyncio.run(official_style_record())