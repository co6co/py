
import cv2
from co6co.utils import log
import argparse
import urllib.parse as urlparse


def capture_dev_image(video_path, output_image_path: str):
    """
    从视频文件中捕获一帧并保存为图片。
    :param video_path: 视频文件路径。
    :param output_image_path: 输出图片文件路径（不能包含中文路径）。
    """
    # 视频文件路径
    try:
        print("开始执行...", video_path)
        # 打开视频文件
        # 给地地址不通，会阻塞在这里
        cap = cv2.VideoCapture(video_path)
        urlparse.quote(video_path)
        if not cap.isOpened():
            print("无法打开视频文件！")
            return
        # 读取一帧
        ret, frame = cap.read()
        if ret:
            # 将中文路径转换为字节类型
            # result = cv2.imwrite(output_image_path, frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            with open(output_image_path, 'wb') as f:
                f.write(buffer)
            print("图片已保存到:", output_image_path)
        else:
            print("无法读取视频帧！", video_path)

        # 释放资源
        cap.release()
    except Exception as e:
        log.err(f"执行 ERROR", e)
        print(f"发生{video_path},错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频文件中捕获一帧并保存为图片。")
    parser.add_argument("-v", "--video_path",    type=str, help="视频文件路径")
    parser.add_argument("-o", "--output_image_path",   type=str, help="输出图片文件路径")
    args = parser.parse_args()
    video_path = args.video_path
    output_image_path = args.output_image_path
    if not video_path or not output_image_path:
        parser.print_help()
    else:
        capture_dev_image(video_path, output_image_path)
