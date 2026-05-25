import os
import sys
from PIL import Image

def combine_images(root_dir, output_dir):
    """
    遍历根目录下的子文件夹，排除含“违法”的文件夹，
    将每个子文件夹内的图片垂直拼接成一张图片，并保存到输出目录的对应位置。
    """
    # 支持的图片扩展名
    SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for root, dirs, files in os.walk(root_dir):
        # 排除文件夹名包含“违法”的文件夹
        dirs[:] = [d for d in dirs if "违法" not in d]

        # 获取当前文件夹相对于根目录的相对路径
        rel_path = os.path.relpath(root, root_dir)
        out_path = os.path.join(output_dir, rel_path)

        # 收集当前文件夹中的图片文件
        image_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(SUPPORTED_EXT)
        ]

        if not image_files:
            continue

        # 按文件名排序，确保顺序稳定
        image_files.sort()

        # 打开图片并统一转换为 RGB 模式
        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"⚠️ 无法打开图片 {img_path}: {e}")

        if not images:
            continue

        # 计算总高度和最大宽度
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)

        # 创建新画布
        combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        # 垂直拼接图片
        y_offset = 0
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height

        # 确保输出目录存在
        os.makedirs(out_path, exist_ok=True)

        # 使用子文件夹名称作为文件名
        folder_name = os.path.basename(root)
        output_file = os.path.join(out_path, f"{folder_name}.jpg")

        # 保存为 JPEG，质量设为 95
        combined_img.save(output_file, 'JPEG', quality=95)
        print(f"✅ 已生成: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python combine_images.py <根文件夹路径> <输出文件夹路径>")
        sys.exit(1)

    root_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(root_directory):
        print(f"❌ 错误: 根文件夹 '{root_directory}' 不存在。")
        sys.exit(1)

    combine_images(root_directory, output_directory)
    print("🎉 全部完成！")