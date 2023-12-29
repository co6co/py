from co6co.utils import log
import os
import cv2
import datetime

 # 按指定图像大小调整尺寸
def resize_image( imreadImage, height = 208, width = 117):
    top, bottom, left, right = (0,0,0,0) 
    # 获取图片尺寸
    h, w, _ = imreadImage.shape
    
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h,w)
    
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。
    
    # RGB颜色
    BLACK = [0,0,0]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv2.copyMakeBorder(imreadImage, top, bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))


async def screenshot( videoPathOrStreamUrl: str, w: int = 208, h: int = 117, isFile: bool = True) -> str:
    """
    视频截图
    视频第一帧作为 poster
    """
    if (isFile and os.path.exists(videoPathOrStreamUrl)) or videoPathOrStreamUrl:
        try:
            cap = cv2.VideoCapture(videoPathOrStreamUrl,cv2.CAP_FFMPEG)  # 打开视频
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
            # cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC , 5)
            ret, fram = cap.read()
            s = None
            if ret:
                if not os.path.exists("tmp"): os.makedirs("tmp")
                s = 'tmp/frame_%s.jpg' % datetime.datetime.now().strftime('%H%M%S%f')
                fram = resize_image(fram, w, h)
                cv2.imwrite(s, fram)
                return s
        finally:
            cap.release()
    return None
