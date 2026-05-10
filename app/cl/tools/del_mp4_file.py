import os
import re
import subprocess
import json
from collections import defaultdict
from co6co.utils import log

def get_video_duration_ffmpeg(file_path):
    """
    使用 ffprobe 获取视频时长（不需要 moviepy，永不报错）
    """
    try:
        cmd = [
            'E:\\Tools\\VXL\\ffmpeg\\bin\\ffprobe',
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            file_path
        ]
       
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        data = json.loads(result.stdout)
        duration = float(data["streams"][0]["duration"])
        return round(duration, 2)
    except:
        return None

def clean_duplicate_videos(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
    if not video_files:
        print("✅ 文件夹中没有 MP4 文件")
        return
   
    file_groups = defaultdict(list)
    pattern = re.compile(r'^(.+?)(_\d+)?\.mp4$', re.IGNORECASE)

    for filename in video_files:
        match = pattern.match(filename)
        if match:
            base = match.group(1)
            full = os.path.join(folder_path, filename)
            file_groups[base].append((filename, full))

    for base, files in file_groups.items():
        if len(files) < 2:
            continue

        print(f"\n======== 处理：{base} ========")
        valid = []

        for name, path in files:
            dur =int(  get_video_duration_ffmpeg(path))
            log.warn("时长",dur)
            if dur is None:
                continue
            size = os.path.getsize(path)
            valid.append((name, path, dur, size))

        if len(valid) < 2:
            print("✅ 本组无重复")
            continue

        dur_groups = defaultdict(list)
        for v in valid:
            dur_groups[v[2]].append(v)

        for dur, items in dur_groups.items():
            if len(items) < 2:
                continue

            print(f"⏱️ 时长相同：{dur}秒，共 {len(items)} 个")
            items_sorted = sorted(items, key=lambda x: x[3])
            log.warn(items_sorted)
            keep = items_sorted[0]
            delete_list = items_sorted[1:]

            keep_name, keep_path, _, _ = keep
            target_name = f"{base}.mp4"
            target_path = os.path.join(folder_path, target_name) 
            # 删除其他
            for del_item in delete_list:
                del_path = del_item[1]
                try:
                    os.remove(del_path)
                    print(f"🗑️ 已删除：{os.path.basename(del_path)}")
                except:
                    print(f"❌ 删除失败：{del_path}")
             # 保留最小文件并重命名
            if keep_path != target_path and not os.path.exists(target_path):
                os.rename(keep_path, target_path)
                print(f"✅ 保留{keep_path}并重命名：{target_name}")


    print("\n🎉 全部处理完成！")

if __name__ == "__main__":
    # 改成你的文件夹路径 
    import argparse 
    
    parser = argparse.ArgumentParser(description="删除重复文件[abc.mp4,abc_1.mp4,abc_2.mp4],且时长相同的文件")
    parser.add_argument("-d", "--dir", type=str, default=None, help="输入mp4所在文件夹")
    args=parser.parse_args()
    if args.dir==None:
        parser.print_help()
    else: 
        if not os.path.isdir(args.dir):
            print("❌ 文件夹不存在")
        else:
            clean_duplicate_videos(args.dir)