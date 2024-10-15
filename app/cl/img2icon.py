from PIL import Image
import argparse


def convert_image_to_icon(image_path, icon_path, sizes=[(256, 256)]):
    """
    将图片转换为图标文件 (.ico)。

    :param image_path: 输入图片的路径
    :param icon_path: 输出图标文件的路径
    :param sizes: 图标的尺寸列表，默认为 [(256, 256)]
    """
    # 打开图片
    img = Image.open(image_path)

    # 转换为图标文件
    img.save(icon_path, format='ICO', sizes=sizes)

    print(f"Icon saved to {icon_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image 转ICON")
    parser.add_argument("-f", "--path",  type=str, help="Image文件路径")
    parser.add_argument("-o", "--output",  type=str, default="icon.ico", help="输出文件")
    parser.add_argument("-w", "--width",  type=int, default="256", help="输出文件")
    args = parser.parse_args()

    if args.path == None:
        parser.print_help()
    else:
        # 示例使用
        icon_path = 'icon.ico'  # 输出图标文件的路径
        convert_image_to_icon(args.path, args.output, [(args.width, args.width)])
