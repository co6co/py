
from sanic.response import text
from sanic import Request
from sanic.response import file, file_stream,json
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from ..base_view import AuthMethodView
from co6co.utils import find_files
from ...model.filters.file_param import FileParam
import os,shutil
import zipfile
import datetime
from pathlib import Path
from co6co .utils import  getRandomStr
from co6co.utils import log 
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

    def __init__(self, root, name ): 
        self.name = name 
        self.path = os.path.join(root,   name)
        self.path=os.path.abspath(self.path) 
        self.isFile=os.path.isfile(self.path)
        if self.isFile:
            self.size = os.path.getsize(self.path)
        self.right = None
        self.updateTime = datetime.datetime.fromtimestamp(os.path.getmtime(self.path))
        pass


class FileViews(AuthMethodView):
    def get_folder_size(self,folder_path):
        """Return the total size of a folder in bytes using pathlib."""
        total_size = 0
        for path in Path(folder_path).rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
    def compress_folder_with_shutil(self,folder_path, output_filename, format='zip'):
        """Compress a folder to a specified format using shutil.make_archive."""
        # 确保输出文件名不包含扩展名
        base_name = os.path.splitext(output_filename)[0] 
        # 创建压缩文件
        shutil.make_archive(base_name, format, folder_path) 
        return os.path.join(folder_path,base_name,format) 

    def zip_directory(self,folder_path, output_zip):
        """
        将给定的文件夹压缩成ZIP文件。
        
        :param folder_path: 要压缩的文件夹路径
        :param output_zip: 输出的ZIP文件路径
        """
        log.warn(folder_path,output_zip)
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
           
            for root, dirs, files in os.walk("I:\\BIT\\ab\\模块1"):
                for file in files:
                    # 计算文件的完整路径
                    file_path = os.path.join(root, file)
                    # 在ZIP文件中保持原来的目录结构
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
    async def head(self, request: Request): 
        args=self.usable_args(request)
        filePath=args.get("path")   
        #return await file(filePath, filename=fileName)  # 使用 file 适用于较小的文件
        response = json({})   
        size=os.path.getsize(filePath) if os.path.isfile(filePath) else self.get_folder_size(filePath)
        response.headers.update( {"Accept-Ranges": "bytes","Content-Length":size,"Content-Type": "application/octet-stream"}) 
        return response


    async def get(self, request: Request): 
        try:
            zipFilePath=None
            args=self.usable_args(request)
            filePath=args.get("path")

            if os.path.isfile(filePath) :
                fileName=os.path.basename(filePath)
            else:
                fileName=getRandomStr(10)+".zip"
                zipFilePath=os.path.join('.', fileName)
                self.zip_directory(filePath ,zipFilePath)
            #return await file(filePath, filename=fileName)  # 使用 file 适用于较小的文件
            return await file_stream(filePath, filename=fileName, chunk_size=4096) 
        finally:
            if zipFilePath !=None:
                os.remove(zipFilePath)



    async def post(self, request: Request):
        """
        列表
        """
        param = FileParam()
        param.__dict__.update(request.json)
        if param.root == None:
            param.root = "/" 
        if param.root.endswith(":"):param.root=param.root+os.sep
        def filter(x): return param.name == None or param.name in x 
        list=os.listdir(param.root) 
        result=[]
        for s in list: 
            if filter(s):
                folder = File(param.root, s )
                result.append(folder) 
        return self.response_json(Result.success({"root": param.root, "res": result}))
    
