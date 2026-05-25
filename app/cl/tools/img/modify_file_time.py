import os
import sys
import datetime
import random
import subprocess
import argparse

def modify_file_times_simple(root_dir: str, start_datetime: datetime.datetime,
    min_minutes: int = 5,
    max_minutes: int = 10):
    """使用PowerShell修改文件时间（最稳定）"""
    current_dt = start_datetime
    
    # 存储每个目录的最早文件时间
    dir_earliest_time = {}

    for root, dirs, files in os.walk(root_dir):
        # 初始化该目录的最早时间
        dir_earliest_time[root] = None
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 随机增加5-10分钟
                minutes_to_add = random.randint(min_minutes*60, max_minutes*60)
                current_dt += datetime.timedelta(seconds=minutes_to_add)
                
                # 格式化时间字符串
                time_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                if os.name == "nt":  # Windows
                    # 使用PowerShell修改创建时间、修改时间和访问时间
                    ps_script = f'''
                    $file = Get-Item "{file_path}"
                    $file.CreationTime = "{time_str}"
                    $file.LastWriteTime = "{time_str}"
                    $file.LastAccessTime = "{time_str}"
                    '''
                    subprocess.run(['powershell', '-Command', ps_script], 
                                 capture_output=True, shell=False)
                else:  # Linux/macOS
                    ts = current_dt.timestamp()
                    os.utime(file_path, (ts, ts))
                
                # 记录该目录下的最早文件时间
                if dir_earliest_time[root] is None or current_dt < dir_earliest_time[root]:
                    dir_earliest_time[root] = current_dt
                
                print(f"✅ {file_path} → {time_str}")
            except Exception as e:
                print(f"❌ 失败 {file_path}: {str(e)[:50]}")
    
    # 修改文件夹时间（早于或等于最早文件时间）
    print("\n📁 修改文件夹时间...")
    for root, dirs, _ in os.walk(root_dir, topdown=False):  # 从最深目录开始
        # 获取该目录下最早的文件时间
        earliest_file_time = dir_earliest_time.get(root)
        
        if earliest_file_time:
            # 文件夹时间设置为最早文件时间减去1秒（确保早于文件）
            folder_time = earliest_file_time - datetime.timedelta(seconds=1)
            time_str = folder_time.strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                if os.name == "nt":  # Windows
                    ps_script = f'''
                    $dir = Get-Item "{root}"
                    $dir.CreationTime = "{time_str}"
                    $dir.LastWriteTime = "{time_str}"
                    $dir.LastAccessTime = "{time_str}"
                    '''
                    subprocess.run(['powershell', '-Command', ps_script], 
                                 capture_output=True, shell=False)
                else:  # Linux/macOS
                    ts = folder_time.timestamp()
                    os.utime(root, (ts, ts))
                
                print(f"📁 {root} → {time_str}")
            except Exception as e:
                print(f"❌ 文件夹失败 {root}: {str(e)[:50]}")
        
        # 处理子目录
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_path in dir_earliest_time:
                earliest_file_time = dir_earliest_time[dir_path]
                if earliest_file_time:
                    folder_time = earliest_file_time - datetime.timedelta(seconds=1)
                    time_str = folder_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    try:
                        if os.name == "nt":  # Windows
                            ps_script = f'''
                            $dir = Get-Item "{dir_path}"
                            $dir.CreationTime = "{time_str}"
                            $dir.LastWriteTime = "{time_str}"
                            $dir.LastAccessTime = "{time_str}"
                            '''
                            subprocess.run(['powershell', '-Command', ps_script], 
                                         capture_output=True, shell=False)
                        else:  # Linux/macOS
                            ts = folder_time.timestamp()
                            os.utime(dir_path, (ts, ts))
                        
                        print(f"📁 {dir_path} → {time_str}")
                    except Exception as e:
                        print(f"❌ 文件夹失败 {dir_path}: {str(e)[:50]}")

def parse_datetime(dt_str: str) -> datetime.datetime:
    """解析日期时间字符串"""
    return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description='批量修改文件和文件夹时间')
    parser.add_argument('root_directory', help='要处理的根目录路径')
    parser.add_argument('-start', '--start-time', required=True,
                       help='开始时间，格式：YYYY-MM-DD HH:MM:SS (例如：2025-02-02 12:12:12)')
    parser.add_argument('-min', '--min-minutes', type=int, default=5,
                       help='最小间隔分钟数 (默认: 5)')
    parser.add_argument('-max', '--max-minutes', type=int, default=10,
                       help='最大间隔分钟数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 验证目录是否存在
    if not os.path.isdir(args.root_directory):
        print(f"❌ 错误：目录 '{args.root_directory}' 不存在")
        sys.exit(1)
    
    # 解析开始时间
    try:
        start_time = parse_datetime(args.start_time)
    except ValueError:
        print(f"❌ 错误：时间格式不正确，请使用 YYYY-MM-DD HH:MM:SS 格式")
        sys.exit(1)
    
    # 验证参数
    if args.min_minutes <= 0 or args.max_minutes <= 0:
        print("❌ 错误：间隔分钟数必须大于0")
        sys.exit(1)
    
    if args.min_minutes > args.max_minutes:
        print("❌ 错误：最小间隔不能大于最大间隔")
        sys.exit(1)
    
    print(f"📁 目标目录: {args.root_directory}")
    print(f"🕑 开始时间: {start_time}")
    print(f"⏱ 间隔范围: {args.min_minutes} ~ {args.max_minutes} 分钟")
    print("-" * 50)
    
    # 执行修改
    modify_file_times_simple(
        root_dir=args.root_directory,
        start_datetime=start_time,
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes
    )
    
    print("\n✅ 所有操作完成！")

if __name__ == "__main__":
    main()