import tempfile
import os
from co6co.utils.File import File
from co6co.utils import async_iterator
import asyncio
import shutil


def writeTempFile():
    # 创建一个临时文件用于保存的数据
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # 定义一个异步生成器来处理文件流
        async def process_data(stream):
            async for data in stream:
                temp_file.write(data)
            print("写文件完成。")
        # 将同步生成器--->异步生成器
        gen = File.readBytes("C:\\Users\\Administrator\\Downloads\\未命名.json", 1)
        asyncio.run(process_data(async_iterator(gen)))
    finally:
        # 确保无论如何都会执行的清理代码
        temp_file.close()
        print("需要关闭后才能复制文件..")
        shutil.copy(temp_file.name, '.\\temp.json')
        # 如果需要，在这里删除临时文件
        print(temp_file.name)
        os.unlink(temp_file.name)  # 删除临时文件


writeTempFile()
