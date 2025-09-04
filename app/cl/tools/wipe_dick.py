import os
import sys
import random
import platform
import ctypes
from ctypes import wintypes

# 仅支持Windows系统
if platform.system() != "Windows":
    print("错误: 此工具仅支持Windows系统")
    sys.exit(1)

# Windows API 常量和函数
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)


def is_admin():
    """检查程序是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def get_disk_size(drive):
    """获取磁盘大小（字节）"""
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

    return total_bytes.value


def wipe_disk(drive, passes=3):
    """
    安全擦除磁盘

    参数:
        drive: 磁盘路径，如 "C:" 或 "\\\\.\\PhysicalDrive0"
        passes: 覆盖次数
    """
    try:
        # 打开磁盘设备
        h_device = kernel32.CreateFileW(
            ctypes.c_wchar_p(drive),
            GENERIC_READ | GENERIC_WRITE,
            0,  # 不共享
            None,
            OPEN_EXISTING,
            0,
            None
        )

        if h_device == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        # 获取磁盘大小
        disk_size = get_disk_size(drive)
        print(f"磁盘大小: {disk_size / (1024**3):.2f} GB")

        # 每次写入的块大小 (100MB)
        block_size = 100 * 1024 * 1024
        total_blocks = (disk_size + block_size - 1) // block_size

        # 多次覆盖磁盘
        for pass_num in range(1, passes + 1):
            print(f"\n第 {pass_num}/{passes} 次覆盖...")

            # 生成填充数据 (最后一次使用0填充)
            if pass_num == passes:
                fill_data = b'\x00' * block_size
            else:
                fill_data = random.randbytes(block_size)

            # 定位到磁盘开始处
            kernel32.SetFilePointerEx(h_device, 0, None, 0)

            # 写入数据
            for block_num in range(total_blocks):
                # 计算当前块的实际大小（最后一块可能较小）
                current_block_size = min(block_size, disk_size - block_num * block_size)

                # 如果是最后一块，调整数据大小
                if current_block_size < block_size:
                    if pass_num == passes:
                        data = b'\x00' * current_block_size
                    else:
                        data = random.randbytes(current_block_size)
                else:
                    data = fill_data

                # 写入数据
                bytes_written = wintypes.DWORD()
                if not kernel32.WriteFile(
                    h_device,
                    data,
                    len(data),
                    ctypes.pointer(bytes_written),
                    None
                ):
                    raise ctypes.WinError(ctypes.get_last_error())

                if bytes_written.value != len(data):
                    raise IOError(f"写入失败，仅写入 {bytes_written.value}/{len(data)} 字节")

                # 显示进度
                progress = (block_num + 1) / total_blocks * 100
                sys.stdout.write(f"\r进度: {progress:.1f}%")
                sys.stdout.flush()

            # 刷新缓存到磁盘
            kernel32.FlushFileBuffers(h_device)

        # 关闭设备
        kernel32.CloseHandle(h_device)
        print("\n\n磁盘擦除完成！数据已无法恢复。")
        return True

    except Exception as e:
        print(f"\n操作失败: {str(e)}")
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
    工作原理：通过多次向磁盘写入随机数据（最后一次使用 0 填充），彻底覆盖原有的数据，使其无法被恢复。
    主要功能：
    1. 列出系统中所有可用的驱动器
    2. 允许用户选择要擦除的驱动器
    3. 支持自定义覆盖次数（推荐 3-7 次）
    4. 显示擦除进度
    5. 确保数据真正写入磁盘（通过 FlushFileBuffers）
    使用注意事项：
    1. 必须以管理员权限运行，否则无法访问磁盘设备
    2. 操作非常危险，会永久删除目标磁盘上的所有数据
    3. 请务必确认选择了正确的磁盘，避免误操作
    4. 对于 SSD 固态硬盘，除了使用此工具，还应确保启用了 TRIM 功能
    使用方法：
    以管理员身份运行 Python
    1. 执行脚本：python disk_wipe.py
    2. 按照提示选择要擦除的驱动器
    3. 输入 YES 确认操作
    4. 选择覆盖次数（默认 3 次）
    5. 等待擦除完成
    """
    # 检查是否以管理员权限运行
    if not is_admin():
        print("错误: 此工具需要以管理员权限运行！")
        print("请右键点击Python，选择'以管理员身份运行'")
        sys.exit(1)

    print("=== 硬盘安全擦除工具 ===")
    print("警告: 此操作将永久删除指定磁盘上的所有数据，无法恢复！")
    print("请确保您知道自己在做什么，并已备份重要数据。\n")

    # 列出可用驱动器
    print("可用的驱动器:")
    drives = list_drives()
    for i, drive in enumerate(drives):
        try:
            size = get_disk_size(drive)
            size_gb = size / (1024**3)
            print(f"{i+1}. {drive} - {size_gb:.2f} GB")
        except:
            print(f"{i+1}. {drive} - 无法获取大小信息")

    # 让用户选择驱动器
    try:
        choice = int(input("\n请输入要擦除的驱动器编号: ")) - 1
        if choice < 0 or choice >= len(drives):
            print("无效的选择")
            sys.exit(1)

        drive = drives[choice]
        confirm = input(f"您确定要擦除 {drive} 上的所有数据吗？这将永久删除所有内容！(输入 YES 确认): ")

        if confirm.upper() != "YES":
            print("操作已取消")
            sys.exit(0)

        # 选择覆盖次数
        passes = input("请输入覆盖次数 (推荐3-7次，默认3次): ")
        passes = int(passes) if passes and passes.isdigit() else 3

        if passes < 1:
            print("覆盖次数必须至少为1")
            sys.exit(1)

        # 执行擦除
        wipe_disk(f"\\\\.\\{drive[:2]}", passes)

    except ValueError:
        print("无效的输入")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
