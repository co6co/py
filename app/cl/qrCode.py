# pip install pyzbar opencv-python
import argparse

import cv2
from pyzbar.pyzbar import decode


def readQRCode(imgPath: str):
    # 读取图像
    print(imgPath)
    image = cv2.imread(imgPath)
    # 图像预处理（根据需要进行预处理）
    # 这里只是简单的示例，实际预处理可能需要更多步骤和参数调整

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Pyzbar 进行二维码解码
    decoded_objects = decode(gray_image)
    # 结果
    result = []
    for obj in decoded_objects:
        data = {"Type": None, "Data": None}
        data["Type"] = obj.type
        data["Data"] = obj.data.decode('utf-8')
        result.append(data)
    if len(result) == 1:
        return result[0]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QR CODE service.")
    parser.add_argument("-i", "--imgPath",  type=str, help="二维码路径")
    args = parser.parse_args()
    if args.imgPath:
        result = readQRCode(args.imgPath)
        if type(result) == list:
            print(*result)
        else:
            print(result)
    else:
        parser.print_help()
