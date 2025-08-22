import os
import sys
import random
import platform
import ctypes
from ctypes import wintypes
import tempfile

# 仅支持Windows系统
if platform.system() != "Windows":
    print("错误: 此工具仅支持Windows系统")
    sys.exit(1)

# Windows API 常量
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)


def is_admin():
    """检查程序是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def get_free_space(drive):
    """获取磁盘可用空间（字节）"""
    free_bytes = wintypes.ULARGE_INTEGER()
    total_bytes = wintypes.ULARGE_INTEGER()
    free_bytes_user = wintypes.ULARGE_INTEGER()

    if not kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive),
        ctypes.pointer(free_bytes),
        ctypes.pointer(total_bytes),
        ctypes.pointer(free_bytes_user)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    return free_bytes.value


def wipe_free_space(drive, passes=3):
    """
    擦除磁盘上的空闲空间（已删除文件的残留区域）

    参数:
        drive: 磁盘路径，如 "C:"
        passes: 覆盖次数
    """
    try:
        # 获取可用空间大小
        free_space = get_free_space(drive)
        if free_space < 1024 * 1024:  # 小于1MB的空间不处理
            print(f"可用空间小于1MB，无需擦除")
            return True

        print(f"磁盘 {drive} 可用空间: {free_space / (1024**3):.2f} GB")

        # 每次写入的块大小 (100MB)
        block_size = 100 * 1024 * 1024

        # 在目标驱动器上创建临时文件
        temp_file_path = os.path.join(drive, "temp_wipe_file.tmp")

        for pass_num in range(1, passes + 1):
            print(f"\n第 {pass_num}/{passes} 次覆盖...")

            try:
                # 打开临时文件
                with open(temp_file_path, "wb") as f:
                    remaining = free_space

                    while remaining > 0:
                        # 计算当前块大小
                        current_block_size = min(block_size, remaining)

                        # 生成填充数据 (最后一次使用0填充)
                        if pass_num == passes:
                            data = b'\x00' * current_block_size
                        else:
                            data = random.randbytes(current_block_size)

                        # 写入数据
                        f.write(data)

                        # 强制刷新到磁盘
                        f.flush()
                        os.fsync(f.fileno())

                        remaining -= current_block_size

                        # 显示进度
                        progress = (free_space - remaining) / free_space * 100
                        sys.stdout.write(f"\r进度: {progress:.1f}%")
                        sys.stdout.flush()

                print()  # 换行

            finally:
                # 删除临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        print("\n\n磁盘空闲空间擦除完成！已删除的文件无法恢复。")
        return True

    except Exception as e:
        print(f"\n操作失败: {str(e)}")
        # 确保临时文件被删除
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        return False


def list_drives():
    """列出所有可用的磁盘驱动器"""
    drives = []
    bitmask = kernel32.GetLogicalDrives()
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if bitmask & 1:
            drives.append(f"{letter}:\\")
        bitmask >>= 1
    return drives


def main():
    """
    工作原理：
    1. 在目标磁盘上创建一个临时文件
    2. 向该文件写入随机数据（或 0），直到填满所有可用空间
    3. 这样就覆盖了所有已删除文件曾经占用的磁盘区域
    4. 最后删除这个临时文件
    主要优势：
    1. 只擦除空闲空间，不会影响当前存在的文件
    2. 操作完成后磁盘可用空间保持不变
    3. 可有效防止通过数据恢复软件找回已删除的文件
    使用注意事项：
    1. 需要管理员权限运行
    2. 操作过程中会暂时占用目标磁盘的全部可用空间
    3. 操作时间取决于可用空间大小和选择的覆盖次数
    4. 对于 SSD，建议使用较少的覆盖次数（1-2 次）以减少写入损耗
    使用方法：
    1. 以管理员身份运行 Python
    2. 执行脚本：python wipe_free_space.py
    3. 选择要擦除空闲空间的驱动器
    4. 输入 YES 确认操作
    5. 选择覆盖次数（默认 1 次）
    6. 等待操作完成
    """
    # 检查是否以管理员权限运行
    if not is_admin():
        print("错误: 此工具需要以管理员权限运行！")
        print("请右键点击Python，选择'以管理员身份运行'")
        sys.exit(1)

    print("=== 磁盘空闲空间擦除工具 ===")
    print("警告: 此操作将覆盖磁盘上所有已删除文件占用的空间，使其无法恢复！")
    print("操作过程中可能会暂时占用大量磁盘空间。\n")

    # 列出可用驱动器
    print("可用的驱动器:")
    drives = list_drives()
    for i, drive in enumerate(drives):
        try:
            free_space = get_free_space(drive)
            free_gb = free_space / (1024**3)
            print(f"{i+1}. {drive} - 可用空间: {free_gb:.2f} GB")
        except:
            print(f"{i+1}. {drive} - 无法获取空间信息")

    # 让用户选择驱动器
    try:
        choice = int(input("\n请输入要擦除空闲空间的驱动器编号: ")) - 1
        if choice < 0 or choice >= len(drives):
            print("无效的选择")
            sys.exit(1)

        drive = drives[choice]
        confirm = input(f"您确定要擦除 {drive} 上的空闲空间吗？(输入 YES 确认): ")

        if confirm.upper() != "YES":
            print("操作已取消")
            sys.exit(0)

        # 选择覆盖次数
        passes = input("请输入覆盖次数 (推荐1-3次，默认1次): ")
        passes = int(passes) if passes and passes.isdigit() else 1

        if passes < 1:
            print("覆盖次数必须至少为1")
            sys.exit(1)

        # 执行擦除
        wipe_free_space(drive, passes)

    except ValueError:
        print("无效的输入")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
