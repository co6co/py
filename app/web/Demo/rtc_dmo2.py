import asyncio
import sys
from datetime import datetime
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
import json
import argparse
import logging
async def simple_record_rtsp(rtsp_url, output_file, duration=30):
    """
    简化的RTSP录制函数
    
    参数:
        rtsp_url: RTSP流地址
        output_file: 输出文件路径
        duration: 录制时长（秒）
    """
    print(f"开始从 {rtsp_url} 录制 {duration} 秒到 {output_file}")
    # 创建对等连接
    pc = RTCPeerConnection()
    # 创建RTSP媒体播放器
    player = MediaPlayer(
        rtsp_url,
        format="rtsp",
        options={
            "rtsp_transport": "tcp",  # 使用TCP，更稳定
        }
    )
    await asyncio.sleep(2)
    if player.video or player.audio:
        print(f"✓ 连接成功，获取到轨道:")
    elif player.video:
        print(f"  - 视频轨道")
    elif player.audio:
        print(f"  - 音频轨道")
    
    else:
        print("  ✗ 未获取到轨道，尝试下一个选项...")
        sys.exit(1)
    # 创建录制器
    recorder = MediaRecorder(output_file)
    
    # 开始录制
    await recorder.start()
    
    # 添加轨道到录制器
    if player.video:
        pc.addTrack(player.video)
        recorder.addTrack(player.video)
        await recorder.addTrack(player.video)
        print("已添加视频轨道")
    
    if player.audio:
        pc.addTrack(player.audio)
        recorder.addTrack(player.audio)
        print("已添加音频轨道")
    print("创建WebRTC连接...")
    await pc.setLocalDescription(await pc.createOffer())
    # 等待指定时间
    print(f"录制中，请等待 {duration} 秒...")
    await asyncio.sleep(duration)
    
    # 停止录制
    await recorder.stop()
    await pc.close()
    print(f"录制完成！文件保存为: {output_file}")

if __name__ == "__main__":
     
    async def test_simple():
         await simple_record_rtsp(
             rtsp_url="rtsp://admin:123456@192.168.3.1/media/video",
             output_file="output.mp4",
             duration=10
         )

    import traceback
    try:
        asyncio.run(test_simple())
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        