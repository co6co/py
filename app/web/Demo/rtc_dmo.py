import asyncio
import sys
from datetime import datetime
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
import json
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_codec(pc, sender, forced_codec):
    """强制使用特定编码器（可选）"""
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

class RTSPRecorder:
    def __init__(self, rtsp_url, output_file=None, duration=30, 
                 video_codec=None, audio_codec=None):
        """
        初始化RTSP录制器
        
        参数:
            rtsp_url: RTSP流地址
            output_file: 输出文件路径（默认自动生成时间戳）
            duration: 录制时长（秒）
            video_codec: 强制使用视频编码器，如"video/H264"
            audio_codec: 强制使用音频编码器，如"audio/opus"
        """
        self.rtsp_url = rtsp_url
        self.duration = duration
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        
        # 如果没有指定输出文件，自动生成
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"recording_{timestamp}.mp4"
        else:
            self.output_file = output_file
        
        self.pc = None
        self.player = None
        self.recorder = None
        
    async def start_recording(self):
        """开始录制RTSP流"""
        try:
            logger.info(f"开始录制 RTSP 流: {self.rtsp_url}")
            logger.info(f"输出文件: {self.output_file}")
            logger.info(f"录制时长: {self.duration} 秒")
            
            # 创建 RTSP 媒体播放器
            options = {
                "rtsp_transport": "tcp",  # 使用TCP传输，更稳定
                "fflags": "nobuffer",     # 减少缓冲延迟
                "flags": "low_delay"
            }
            
            self.player = MediaPlayer(
                self.rtsp_url,
                format="rtsp",
                options=options
            )
            
            # 创建录制器
            self.recorder = MediaRecorder(self.output_file,format="mp4")
            
            # 创建对等连接
            self.pc = RTCPeerConnection()
            
            # 从播放器获取音视频轨道
            if self.player.video:
                self.pc.addTrack(self.player.video)
                logger.info("已添加视频轨道")
            
            if self.player.audio:
                self.pc.addTrack(self.player.audio)
                logger.info("已添加音频轨道")
            
            # 设置强制编码器（可选）
            if self.video_codec:
                for sender in self.pc.getSenders():
                    if sender.track and sender.track.kind == "video":
                        force_codec(self.pc, sender, self.video_codec)
                        logger.info(f"强制使用视频编码器: {self.video_codec}")
            
            if self.audio_codec:
                for sender in self.pc.getSenders():
                    if sender.track and sender.track.kind == "audio":
                        force_codec(self.pc, sender, self.audio_codec)
                        logger.info(f"强制使用音频编码器: {self.audio_codec}")
            
            # 启动录制器
            await self.recorder.start()
            logger.info("录制器已启动")
            
            # 创建本地SDP offer
            await self.pc.setLocalDescription(await self.pc.createOffer())
            
            logger.info(f"开始录制，等待 {self.duration} 秒...")
            
            # 等待指定的录制时长
            for i in range(self.duration):
                await asyncio.sleep(1)
                if i % 10 == 0:  # 每10秒打印一次进度
                    logger.info(f"录制中... 已录制 {i+1} 秒")
            
            logger.info("录制完成，正在关闭...")
            
        except Exception as e:
            logger.error(f"录制过程中发生错误: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 停止录制器
            if self.recorder:
                await self.recorder.stop()
                logger.info("录制器已停止")
            
            # 关闭对等连接
            if self.pc:
                await self.pc.close()
                logger.info("对等连接已关闭")
            
            # 关闭播放器
            if self.player:
                # MediaPlayer 可能需要特殊方式关闭
                # 这里我们依赖垃圾回收
                self.player = None
                
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RTSP流录制工具")
    parser.add_argument("--url", help="RTSP流URL地址", required=True)
    parser.add_argument("--output", "-o", help="输出文件路径", default=None)
    parser.add_argument("--duration", "-d", help="录制时长（秒）", type=int, default=30)
    parser.add_argument("--codec", "-c", help="强制视频编码器", default="video/H264")
    
    # 如果通过命令行运行，解析参数
    if len(sys.argv) > 1:
        args = parser.parse_args()
        rtsp_url = args.url
        output_file = args.output
        duration = args.duration
        video_codec = args.codec
    else:
        parser.print_help()
        sys.exit(1) 
    # 创建录制器实例
    recorder = RTSPRecorder(
        rtsp_url=rtsp_url,
        output_file=output_file,
        duration=duration,
        video_codec=video_codec
    )
    
    # 开始录制
    await recorder.start_recording()
    
    logger.info(f"录制完成！文件保存为: {recorder.output_file}")


if __name__ == "__main__":
    asyncio.run(main())