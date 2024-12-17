
from sanic.response import text 
from sanic import Request
from sanic.response import file, file_stream, json,raw
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from ..base_view import AuthMethodView 
from ...model.filters.file_param import FileParam
import os 
import datetime
from pathlib import Path 
from co6co.utils import log
from urllib.parse import quote
import tempfile
from cacheout import Cache
import uuid

class Range:
    def __init__(self,s,e,size,total):
        self.start=s
        self.end=e
        self.size=size
        self.total=total
        pass
    def start(self) -> int:
        return self.start

    def end(self) -> int:
        return self.end

    def size(self) -> int:
        return self.size

    def total(self) -> int:
        return self.total
class File:
    isFile: bool
    name: str
    path: str
    right: str
    date: datetime.datetime
    size: int

    def __init__(self):
        self.isFile = None
        self.name = None
        self.path = None
        self.right = None
        self.updateTime = None
        self.size = None
        pass

    def __init__(self, root, name):
        self.name = name
        self.path = os.path.join(root,   name)
        self.path = os.path.abspath(self.path)
        self.isFile = os.path.isfile(self.path)
        if self.isFile:
            self.size = os.path.getsize(self.path)
        self.right = None
        self.updateTime = datetime.datetime.fromtimestamp(os.path.getmtime(self.path))
        pass


class FolderView(AuthMethodView):
    routePath = "/zip"

    async def head(self, request: Request):
        """
        文件夹打包
        """
        args = self.usable_args(request)
        filePath = args.get("path")
        if os.path.isfile(filePath):
            raise Exception("该方法不支持文件") 
        timeStr = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        fileName = "{}_{}.zip".format(os.path.basename(filePath), timeStr)
        # cache:Cache= request.app.ctx.Cache
        # uid=uuid.uuid4()
        # cache.add(uid,fileName)
        zipFilePath = os.path.join('.', fileName)   
        id=request.headers.get("session")
        log.warn(id, request.app.ctx)
        request.app.ctx.data={id:{filePath:zipFilePath}}
        await self.zip_directory(filePath, zipFilePath) 
        return await self.response_size(filePath=zipFilePath) 

    async def get(self, request: Request):
        try:
            zipFilePath = None
            temp_file = None
            args = self.usable_args(request)
            filePath = args.get("path") 
            id=request.headers.get("session")
            log.warn(id)
            filePath=request.app.ctx.data[id].get(filePath)
            log.warn(filePath)  
            if os.path.isfile(filePath):
                fileName = os.path.basename(filePath)
            headers = self.createContentDisposition(fileName)  
            file_size = os.path.getsize(filePath)
            headers.update({'Content-Length': str(file_size),'Accept-Ranges': 'bytes',})
            #return await file(filePath, headers=headers)  # 使用 file 适用于较小的文件 传送完整文件
            # return await file(filePath,filename= fileName)  # 使用 file 适用于较小的文件 中文名乱码
            range_header = request.headers.get('Range')
            log.warn(range_header)
            if range_header:
                unit, ranges = range_header.split('=')
                if unit != 'bytes':
                    return text('Only byte ranges are supported', status=416)

                start, end = map(lambda x: int(x) if x else None, ranges.split('-'))
                if start is None:
                    start = file_size - end
                    end = file_size - 1
                elif end is None or end >= file_size:
                    end = file_size - 1
            log.warn(file_size,start,end)
            #return await file_stream(filePath,status=206, headers=headers )  # 未执行完 finally 就开始执行
            # 返回二进制数据
            # 读取文件内容为二进制数据
            
            binary_data:bytes=None
            with open(filePath, 'rb') as f:
                log.warn(start,end-start)
                f.seek(start)
                binary_data = f.read(end-start+1)
            return  raw(
                binary_data,status=206, headers=headers 
            )
        finally:
            if zipFilePath != None and os.path.exists(zipFilePath):
                # os.remove(zipFilePath)
                # os.unlink(zipFilePath)
                pass
            if temp_file != None:
                temp_file.close()
                os.unlink(temp_file.name)


class FileViews(AuthMethodView):

    async def head(self, request: Request):
        """
        文件或目录大小
        """
        return await self.response_size(request=request)

    async def get(self, request: Request):
        """
        下载文件
        """
        args = self.usable_args(request)
        filePath = args.get("path")
        if os.path.isfile(filePath):
            pass
        else:
            raise Exception("该方法不支持文件夹下载")
        #headers = self.createContentDisposition(fileName)
        # return await file(filePath,headers=headers )  # 使用 file 适用于较小的文件
        # return await file(filePath,filename= fileName)  # 使用 file 适用于较小的文件 中文名乱码
        return await file_stream(filePath )  # 未执行完 finally 就开始执行

    async def post(self, request: Request):
        """
        列表
        """
        param = FileParam()
        param.__dict__.update(request.json)
        if param.root == None:
            param.root = "/"
        if param.root.endswith(":"):
            param.root = param.root+os.sep

        def filter(x): return param.name == None or param.name in x
        list = os.listdir(param.root)
        result = []
        for s in list:
            if filter(s):
                folder = File(param.root, s)
                result.append(folder)
        return self.response_json(Result.success({"root": param.root, "res": result}))

    async def put(self, request: Request):
        # 创建一个临时文件用于保存上传的数据
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            # 定义一个异步生成器来处理文件流
            async def process_data(stream):
                async for data in stream:
                    temp_file.write(data)

            # 调用 request.stream 并传入生成器
            await request.stream(process_data)
            # 这里可以添加更多处理逻辑，比如保存文件到永久位置等
            return self.response_json(Result.success())

        finally:
            # 确保无论如何都会执行的清理代码
            temp_file.close()
            # 如果需要，在这里删除临时文件
            os.unlink(temp_file.name)  # 删除临时文件
