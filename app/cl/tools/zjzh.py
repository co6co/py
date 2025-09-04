#-- coding: utf-8 --
# 证件照工具

# pip install pillow rembg opencv-python
# pip install onnxruntime
#
# rembg 使用的模型 及场景
# u2net（默认） 通用模型，平衡速度与精度，对常规物体（如人像、产品）效果稳定 通用场景
# u2netp 轻量版 u2net，速度更快但精度略低 对速度要求高的  简单场景
# isnet-anime 基于 Anime 结构的模型，对动漫、插画、高细节图像（如发丝、渐变）更敏感  动漫/插画、高细节人像
# silueta 极简模型，体积小但精度较低 快速测试或低算力设备
# isnet-general-use 通用增强版模型，对复杂背景、透明物体的分割更鲁棒 复杂背景（如玻璃、烟雾）

from PIL import Image
from rembg import remove
import cv2
import numpy as np

# 定义证件照尺寸（毫米）和相纸尺寸（毫米）
SIZE_SPEC = {
    '一寸': (25, 35),
    '小一寸': (22, 32),
    '二寸': (35, 49),
    '小二寸': (33, 48),
    '大一寸': (39, 56),  # 常见大一寸尺寸（如身份证）
}

PAPER_SIZE = {
    '6寸': (102, 152),  # 4英寸×6英寸
    '7寸': (127, 178),  # 5英寸×7英寸
}

DPI = 300  # 打印常用分辨率

def mm_to_pixel(mm, dpi):
    """毫米转像素"""
    return int(round(mm * dpi / 25.4))

def detect_face(image):
    """使用OpenCV检测人脸，返回最大人脸的坐标（x, y, w, h）"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("人脸信息->",faces)
    if not faces.any():
        return None
    # 选择最大的人脸
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, w, h)

def resize_and_crop(image, target_width, target_height):
    """调整图片大小并裁剪到目标尺寸，优先保留人脸区域"""
    orig_width, orig_height = image.size
    target_ratio = target_width / target_height
    orig_ratio = orig_width / orig_height

    # 检测人脸
    face = detect_face(image)
    if face:
        fx, fy, fw, fh = face
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        # 人脸区域扩展系数（可根据需求调整）
        expand_w = fw * 1.5
        expand_h = fh * 2.0
        # 计算裁剪区域（确保不超出原图边界）
        crop_left = max(0, int(face_center_x - expand_w // 2))
        crop_top = max(0, int(face_center_y - expand_h // 2))
        crop_right = min(orig_width, int(face_center_x + expand_w // 2))
        crop_bottom = min(orig_height, int(face_center_y + expand_h // 2))
        # 裁剪人脸区域
        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    else:
        # 无人脸时按宽高比裁剪
        if orig_ratio > target_ratio:
            new_height = orig_height
            new_width = int(orig_height * target_ratio)
            left = (orig_width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = new_height
        else:
            new_width = orig_width
            new_height = int(orig_width / target_ratio)
            left = 0
            top = (orig_height - new_height) // 2
            right = new_width
            bottom = top + new_height
        cropped = image.crop((left, top, right, bottom))

    # 缩放至目标尺寸
    resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized

def change_background(image, bg_color=(255, 255, 255)):
    """去除背景并填充指定颜色"""
    # 去除背景（需要安装rembg库）
    output = remove(image, 
       # model_name="isnet-anime",  # 高细节模型
        alpha_matting=True,         # 启用 Alpha 融合
        alpha_matting_foreground_threshold=220,  # 调小以保留暗部发丝
        alpha_matting_background_threshold=5,    # 调小以保留亮背景细节
        alpha_matting_erode_size=5,             # 调小以减少边缘损失
        post_process_mask=True                  # 后处理优化掩码
    )
    # 创建背景图
    bg = Image.new('RGBA', output.size, bg_color + (255,))  # RGBA模式
    # 合并前景和背景
    combined = Image.alpha_composite(bg, output)
    # 转换为RGB（若原图无透明通道）
    return combined.convert('RGB')

def calculate_layout(photo_size_px, paper_size_mm, margin_mm=2, photo_spacing_mm=2, dpi=DPI):
    """计算排版的行列数及最大容量，包含照片之间的边距"""
    # 转换单位
    margin_px = mm_to_pixel(margin_mm, dpi)
    photo_spacing_px = mm_to_pixel(photo_spacing_mm, dpi)
    paper_width_px = mm_to_pixel(paper_size_mm[0], dpi)
    paper_height_px = mm_to_pixel(paper_size_mm[1], dpi)
    
    # 相纸有效尺寸（减去边缘边距）
    effective_width = paper_width_px - 2 * margin_px
    effective_height = paper_height_px - 2 * margin_px

    # 证件照尺寸
    photo_width_px, photo_height_px = photo_size_px
    
    # 计算行列数，考虑照片之间的边距
    # 每行/列的总宽度/高度 = 照片宽度/高度 + 照片间距（最后一张除外）
    # 正确的计算公式：列数 = 有效宽度 // (照片宽度 + 照片间距) + 1（如果剩余空间足够放一张照片）
    # 修复运算符优先级问题，添加括号
    cols = (effective_width // (photo_width_px + photo_spacing_px)) + \
           (1 if effective_width % (photo_width_px + photo_spacing_px) >= photo_width_px else 0)
    rows = (effective_height // (photo_height_px + photo_spacing_px)) + \
           (1 if effective_height % (photo_height_px + photo_spacing_px) >= photo_height_px else 0)
    
    # 确保至少能放一张照片
    cols = max(1, cols)
    rows = max(1, rows)
    max_count = cols * rows

    return rows, cols, max_count, (photo_width_px, photo_height_px), photo_spacing_px

# 添加调试信息以验证计算结果
def generate_id_photo(
    input_path,
    output_path,
    size_type='一寸',
    paper_size='6寸',
    bg_color=(255, 255, 255),
    margin_mm=2,
    photo_spacing_mm=2,  # 新增参数：照片之间的边距
    count=16
):
    """生成证件照并排版"""
    # 验证输入尺寸
    if size_type not in SIZE_SPEC:
        raise ValueError(f"不支持的证件照尺寸：{size_type}，可选：{list(SIZE_SPEC.keys())}")
    if paper_size not in PAPER_SIZE:
        raise ValueError(f"不支持的相纸尺寸：{paper_size}，可选：{list(PAPER_SIZE.keys())}")

    # 读取原始图片
    original = Image.open(input_path).convert('RGB')

    # 获取目标尺寸（毫米转像素）
    target_mm = SIZE_SPEC[size_type]
    target_px = (
        mm_to_pixel(target_mm[0], DPI),
        mm_to_pixel(target_mm[1], DPI)
    )

    # 调整尺寸并裁剪（优先保留人脸）
    resized_photo = resize_and_crop(original, target_px[0], target_px[1])

    # 替换背景
    if bg_color != (255, 255, 255):
        resized_photo = change_background(resized_photo, bg_color)

    # 计算排版布局，包含照片之间的边距
    paper_size_mm_val = PAPER_SIZE[paper_size]
    rows, cols, max_count, _, photo_spacing_px = calculate_layout(target_px, paper_size_mm_val, margin_mm, photo_spacing_mm, DPI)

    # 添加调试信息
    print(f"调试信息：")
    print(f"- 相纸尺寸: {paper_size_mm_val}mm")
    print(f"- 证件照尺寸: {target_mm}mm ({target_px}px)")
    print(f"- 边缘边距: {margin_mm}mm")
    print(f"- 照片间距: {photo_spacing_mm}mm ({photo_spacing_px}px)")
    print(f"- 计算得到: {rows}行 × {cols}列 = {max_count}张")

    # 检查数量是否超过最大容量
    if count > max_count:
        raise ValueError(f"相纸最多可排版{max_count}张，当前需要{count}张")

    # 生成足够数量的证件照（复制）
    photos = [resized_photo] * count

    # 合并排版，使用照片之间的边距
    collage = create_collage(photos, paper_size_mm_val, margin_mm, photo_spacing_px, DPI, cols)

    # 保存结果
    collage.save(output_path)
    print(f"成功生成证件照，保存至：{output_path}")

def create_collage(photos, paper_size_mm, margin_mm=2, photo_spacing_px=0, dpi=DPI, cols=0):
    """将多张证件照合并到相纸上，包含照片之间的边距"""
    # 相纸像素尺寸
    paper_width_px = mm_to_pixel(paper_size_mm[0], dpi)
    paper_height_px = mm_to_pixel(paper_size_mm[1], dpi)
    collage = Image.new('RGB', (paper_width_px, paper_height_px), (255, 255, 255))  # 白色背景

    margin_px = mm_to_pixel(margin_mm, dpi)
    photo_width, photo_height = photos[0].size
    x, y = margin_px, margin_px

    for i, photo in enumerate(photos):
        collage.paste(photo, (x, y))
        # 更新x坐标，加上照片宽度和间距
        x += photo_width + photo_spacing_px
        if (i + 1) % cols == 0:
            # 换行，重置x坐标，更新y坐标
            x = margin_px
            y += photo_height + photo_spacing_px

    return collage

# 示例用法修改为使用照片间距参数
if __name__ == "__main__": 
    generate_id_photo(
        input_path="C:\\Users\\Administrator\\Desktop\\ddd\\demo.jpg",
        output_path="C:\\Users\\Administrator\\Desktop\\ddd\\output_1inch_6inch.jpg",
        size_type="一寸",
        paper_size="7寸",
        bg_color=(255, 0, 0),  # 白色背景
        margin_mm=1,  # 相纸边缘边距2mm
        photo_spacing_mm=2,  # 照片之间边距2mm
        count=16  # 排版12张
    )