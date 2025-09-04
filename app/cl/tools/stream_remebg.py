
import cv2
import rembg
# 加载视频文件
#video_path = 'input_video.mp4'
video_path = 'rtsp://admin:lanbo8338299@192.168.3.200:554/Streaming/channels/401'
cap = cv2.VideoCapture(video_path)
# 获取视频帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 创建输出视频文件
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# 使用rembg进行抠图
count=100
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 将帧转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 使用rembg进行抠图 - 修复方法名
    mask = rembg.remove(frame_rgb)
    # 将抠图结果转换回BGR格式
    frame_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    # 写入输出视频文件
    out.write(frame_bgr)
    count-=1
    if count<0:
        break   
# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()