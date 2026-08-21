import os
import secrets

s = int(input("输入大小MB："))
fPath = input("输入文件名：")
chunk = 1024 * 1024  # 每次写 1MB


def getData(chunkSize=1024 * 1024):
    return os.urandom(chunkSize) 
    #return secrets.token_bytes( chunkSize )  # secrets 和 os.urandom 底层一样，但语义上更明确是“安全用途”


with open(fPath, "wb") as f:
    # 0x00 填充文件
    #f.truncate(s * 1024 * 1024)  # 文件系统层面直接分配空间（稀疏文件）# 瞬间完成
    for i in range(s): 
        f.write(getData(chunk))
    
