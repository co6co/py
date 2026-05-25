import os
import sys
import re
from PIL import Image

def combine_images_by_group(root_dir, output_dir):
    """
    按文件名前缀分组（如xxx_1.jpg、yyy_2.jpg），
    每组3张横向排列，多组纵向堆叠。
    仅处理包含“违法”的文件夹。
    """
    SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for root, dirs, files in os.walk(root_dir):
        # 仅处理包含“违法”的文件夹（延续您之前的需求）
        if "违法" not in root:
            continue
        
        # 收集所有图片文件
        image_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(SUPPORTED_EXT)
        ]
        
        if not image_files:
            continue
        
        # 按文件名前缀分组（提取“_”前的部分作为组名）
        groups = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # 正则匹配：前缀_数字.扩展名（如xxx_1.jpg → 前缀xxx）
            match = re.match(r'^(.*?)_\d+\.\w+$', filename)
            if match:
                prefix = match.group(1)
                groups.setdefault(prefix, []).append(img_path)
            else:
                print(f"⚠️ 跳过非标准文件名: {filename}")
        
        if not groups:
            print(f"⚠️ 文件夹 {root} 无符合格式图片，跳过")
            continue
        
        # 准备每组的有效图片（每组必须3张，按数字排序）
        valid_groups = []
        for prefix, img_paths in groups.items():
            # 按文件名中的数字排序（确保xxx_1在xxx_2前）
            img_paths.sort(key=lambda x: int(re.search(r'_(\d+)\.', os.path.basename(x)).group(1)))
            
            if len(img_paths) != 3:
                print(f"⚠️ 组 {prefix} 有 {len(img_paths)} 张图片（需3张），跳过")
                continue
            
            # 打开图片并验证
            images = []
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"⚠️ 无法打开 {img_path}: {e}")
            
            if len(images) == 3:
                valid_groups.append(images)
        
        if not valid_groups:
            print(f"⚠️ 文件夹 {root} 无有效组，跳过")
            continue
        
        # 假设所有图片尺寸相同，取第一张的尺寸
        sample_img = valid_groups[0][0]
        img_width, img_height = sample_img.size
        
        # 计算画布大小：宽度=3张×单宽，高度=组数×单高
        total_width = 3 * img_width
        total_height = len(valid_groups) * img_height
        
        # 创建白色背景画布
        combined_img = Image.new('RGB', (total_width, total_height), color='white')
        
        # 拼接图片：每组横向排列，组间纵向堆叠
        y_offset = 0
        for group in valid_groups:
            x_offset = 0
            for img in group:
                combined_img.paste(img, (x_offset, y_offset))
                x_offset += img_width
            y_offset += img_height
        
        # 确保输出目录结构一致
        rel_path = os.path.relpath(root, root_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(out_path, exist_ok=True)
        
        # 输出文件名：子文件夹名_合成.jpg
        folder_name = os.path.basename(root)
        output_file = os.path.join(out_path, f"{folder_name}_合成.jpg")
        
        # 保存为高质量JPEG
        combined_img.save(output_file, 'JPEG', quality=95)
        print(f"✅ 已生成: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python combine_by_group.py <根文件夹路径> <输出文件夹路径>")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    if not os.path.isdir(root_directory):
        print(f"❌ 错误: 根文件夹 '{root_directory}' 不存在。")
        sys.exit(1)
    
    combine_images_by_group(root_directory, output_directory)
    print("🎉 全部完成！")