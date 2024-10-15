import os
import shutil
import argparse
import win32api
import win32con
import ctypes
from pathlib import Path
import psutil

class UP:
    FILE_ATTRIBUTE_READONLY = 0x00000001  # 只读文件。
    FILE_ATTRIBUTE_HIDDEN = 0x00000002  # 隐藏文件。
    FILE_ATTRIBUTE_SYSTEM = 0x00000004  # 系统文件。
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010  # 目录。
    FILE_ATTRIBUTE_ARCHIVE = 0x00000020  # 存档文件。
    FILE_ATTRIBUTE_NORMAL = 0x00000080  # 正常文件，没有其他属性。
    FILE_ATTRIBUTE_TEMPORARY = 0x00000100  # 临时文件。
    FILE_ATTRIBUTE_SPARSE_FILE = 0x00000200  # 稀疏文件。
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400  # 重新解析点（符号链接）。
    FILE_ATTRIBUTE_COMPRESSED = 0x00000800  # 压缩文件。
    FILE_ATTRIBUTE_OFFLINE = 0x00001000  # 脱机文件。
    FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x00002000  # 不包含在内容索引中。
    FILE_ATTRIBUTE_ENCRYPTED = 0x00004000  # 加密文件。
    FILE_ATTRIBUTE_VIRTUAL = 0x00010000  # 虚拟文件。

    usb_drive_path: str = None
    info_content: str = None
    bg_image_path: str = None
    icon_path: str = None
    label: str = None

    def __init__(self,
                 usb_drive_path: str = "E:\\",
                 info_content: str = """
                                    Name: ME
                                    Email: co6co@qq.com
                                    Phone: 1234567890
                                    """,
                 bg_image_path: str = None,
                 icon_path: str = None,
                 label: str = "移动硬盘") -> None:
        self.usb_drive_path = usb_drive_path
        self.info_content = info_content
        self.bg_image_path = bg_image_path
        self.icon_path = icon_path
        self.label = label
        pass

    @staticmethod
    def setFile(*files: str):        
        kernel32 = ctypes.windll.kernel32
        for file in files: 
            kernel32.SetFileAttributesW(file, UP.FILE_ATTRIBUTE_READONLY | UP.FILE_ATTRIBUTE_SYSTEM | UP.FILE_ATTRIBUTE_HIDDEN)

    def write(self):
        # 创建 info.txt 文件
        filePath = os.path.join(self. usb_drive_path, 'info.txt')
        with open(filePath, 'w') as info_file:
            info_file.write(self. info_content)
        UP.setFile(filePath)
        # 创建 Autorun.inf 文件
        autorun_content = """[Autorun] 
icon={}
label={}
""".format(os.path.basename(self.icon_path), self.label)
        filePath = os.path.join(self.usb_drive_path, 'Autorun.inf')
        with open(filePath, 'w') as autorun_file:
            autorun_file.write(autorun_content)
        UP.setFile(filePath)
        # 复制背景图片和图标文件到 U 盘
        path=Path(self.bg_image_path) 
        bg=os.path.join(self.usb_drive_path, "bg{}".format(path.stem, path.suffix))
        path=Path(self.icon_path) 
        icon=os.path.join(self.usb_drive_path, "icon{}".format(path.stem, path.suffix))
        shutil.copy(self.bg_image_path, bg)
        shutil.copy(self.icon_path, icon)
        self.setFile(bg,icon)
        print("U盘设置完成！")
    
    @staticmethod
    def get_usb_info():
        """获取 U 盘信息"""
        partitions = psutil.disk_partitions()
        usb_drives = [p for p in partitions if 'removable' in p.opts]
        for drive in usb_drives:
            usage = psutil.disk_usage(drive.mountpoint)
            print(f"Device: {drive.device}")
            print(f"Mount Point: {drive.mountpoint}")
            print(f"File System: {drive.fstype}")
            print(f"Total Size: {usage.total / (1024**3):.2f} GB")
            print(f"Used Space: {usage.used / (1024**3):.2f} GB")
            print(f"Free Space: {usage.free / (1024**3):.2f} GB")
            print(f"Usage: {usage.percent}%")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="U盘写入 ")
    parser.add_argument("-u", "--upan",  type=str, help="u盘所在路径: U:\\")
    parser.add_argument("-f", "--image",  type=str, help="Image文件路径")
    parser.add_argument("-i", "--icon",  type=str, default="icon.ico", help="U盘图标")
    parser.add_argument("-t", "--text",  type=str, default="Name: John Doe\ntel:1588xxxxxxxx", help="输出文件,换行:`n")
    parser.add_argument("-l", "--label",  type=str, default="本地磁盘", help="U盘卷标")
    args = parser.parse_args()

    if args.upan == None or args.image == None or args.icon == None or args.text == None:
        parser.print_help()
        UP.get_usb_info()
    else:
        # 示例使用
        up = UP(args.upan, args.text, args.image, args.icon, args.label)
        up.write()
