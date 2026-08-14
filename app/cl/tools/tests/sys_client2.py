from asyncio import streams
import os
import re, base64
from shutil import ExecError

import argparse

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


class CreateHandler(FileSystemEventHandler):
    def on_created(self, event):
        # event.is_directory 为 True 表示创建的是文件夹
        src_path:str=event.src_path
        if event.is_directory:
            print(f"新文件夹被创建: {src_path}")  
        else:
            print(f"新文件被创建: {src_path}")
'''
            -----BEGIN PGP MESSAGE-----
Version: GnuPG v1

hQEMA4fLSGZcsDhlAQgAo9U+YNoEwGkoW4E6LMuFt/0xWHogcymlybgcy8ALovXD
uAWVux0vxyIia3q3TntbgSvTW92kJemuq+9CsQYwR3QDMvum+AL0i3z+OagAoCPv
UKbRysijmrEbvnNjCFAI53Vde9/WXH9Xq9RHN7qeQgmdV1S68HEL4R6kDkk/XrtD
u5Xz/3cLpV9FQijvfI5cwkWX+sJaum8nhbC5d+ni6NiYm4C+LpA1kIBHWkBK5f96
a0FnaxFPwI08FC4M4gEZGKLk17PXE1TFnyV6U2qkV0/XCX5SDvj7PTiKxAp4H0u/
Wah7shQBnGj16ZxdOQMGvB4gbFybNFF6pzqCQ7IPQ9LBEwEVN6IsKhKror8DHAms
0rwpwBQwiOgf0RGLuwM216eLOiwrHGL9Jhqc+SW1D/YFv1F1LJ5IdEm/455DY8/U
sUvtOrYagSpUFT1GksAQNhJziwvauBvnaNH0Gg9Two1nvl3SKpdXY0Se9Naw8gIc
MJamg7muQxWls/WBuSjwIMngkyOfI96lkXItE9dzGaJhtEitKLiMwMACucxx6BUb
iCivG3DdK48piZbS7x4sboGX7ZvPl7j2zK/o5gPpUpezwEHNlEv1ch4phVHxzvm4
MgVN9FhhWYP/w0VSkUVTBHn2iSI+TaunwplTuBlpfdElUNjv4H34HbCOdHIza6xm
qwidSUpwH9miXq0Dul/S/nRnJ+YG2DyEifPDjLjvkdhab9fogjQD/kDWHbzjBo4t
V+u+tnek77dEoFTEAswgnixXcASze8ZSAAobv+uszFQWGc2NE2eNiZMjBjEJGR24
4qQMwhXxq7+67P06R1OaNcbEFfzb55g5h8j3fplhqEBfOsiVED7BQNi6tWqDE0Ey
6eKjsdMXM+U0xZw5PUn4gMFmH6zKCBISXrBOLgVyrgAuLKmUFnGetbWCHv17u/ZT
JsEDSo3/fKRw/JWK32EJxoa6gOJO
=BNj6
-----END PGP MESSAGE-----
            '''
            

if __name__ == "__main__":
    watch_path = "/path/to/your/directory"  # ← 改成你要监听的目录
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--watch", type=str, default="C:\\temp", help="监听的目录")
    args = args_parser.parse_args()
    watch_path = args.watch
    observer = Observer()
    observer.schedule(
        CreateHandler(), watch_path, recursive=False
    )  # recursive=True 可监听子目录
    observer.start()

    print(f"正在监听目录: {watch_path} (按 Ctrl+C 退出)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n已停止监听")
    observer.join()
