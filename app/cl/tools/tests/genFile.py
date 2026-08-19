import os

s = input("输入大小MB：")
fPath = input("输入文件名：")
with open(fPath, "wb") as f:
    f.write(os.urandom(int(s) * 1024 * 1024))
