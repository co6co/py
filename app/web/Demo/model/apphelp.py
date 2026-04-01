import os
from pathlib import Path
from typing import AsyncGenerator, Generator, Union
import aiofiles
import asyncio
def get_file_path(fileName ):
    """获取文件路径"""
    print("当前目录", str(Path.cwd()), "apphelp所在目录",os.path.dirname(__file__)) 
    base_dir=str(Path.cwd())
    return  os.path.join(base_dir, 'pages', fileName)  
def read_file_content(file_path: Union[str, Path]):
    """读取HTML文件"""
   
    if not os.path.exists(file_path):
        return f"404 Not Found: {file_path}"  
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

async def read_file_line(file_path: Union[str, Path]) -> AsyncGenerator[str, None]:
    """
    异步流式读取 HTML 文件
    """ 
    if not os.path.exists(file_path):
        yield f"<h1>错误: 文件不存在 - {file_path}</h1>"
        return
    
    try:
        # 使用 aiofiles 异步读取文件
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            # 一次读取一行，避免内存占用过大
            async for line in f:
                yield line
    except Exception as e:
        yield f"<h1>错误: 读取文件失败 - {str(e)}</h1>"

async def read_chunked(file_path: Union[str, Path], chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    """
    分块读取 HTML 文件，控制每次读取的数据量
    """
    if not os.path.exists(file_path):
        yield b""
        return 
    try:
        async with aiofiles.open(file_path, 'rb' ) as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        print(f"读取文件失败: {e}")
        yield b""

async def process_line(html_file_path: Union[str, Path]) -> AsyncGenerator[str, None]:
    """
    处理 HTML 流，可以在传输过程中修改内容
    """
    async for chunk in read_file_line(html_file_path):
        # 这里可以对 HTML 内容进行处理
        # 例如：添加水印、替换内容、压缩等
        newContent = chunk.replace(
            '{{timestamp}}', 
            asyncio.get_event_loop().time()
        )
        yield newContent