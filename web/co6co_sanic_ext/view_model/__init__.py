from functools import wraps
from sanic.views import HTTPMethodView  # 基于类的视图
from sanic.request.form import File  # 基于类的视图
from sanic import Request
from sanic.response import json, raw, BaseHTTPResponse
from co6co.data.result import Result, Page_Result

from co6co.utils.json_util import JSONEncoder
from typing import TypeVar, Dict, List, Any, Tuple
import aiofiles
import os
import multipart
from io import BytesIO
from sqlalchemy import Select
from co6co_db_ext.po import BasePO, UserTimeStampedModelPO
from datetime import datetime
from co6co.utils.tool_util import list_to_tree, get_current_function_name
from co6co.utils import log, getDateFolder
from urllib.parse import quote
import zipfile
from pathlib import Path
from co6co_web_session import Session
from co6co_web_session.base import SessionDict
from typing import Tuple
from co6co.utils import tool_util as utils
from co6co.utils.modules import deprecated
import json as sys_json 

def response_json(data: Result | Page_Result, status: int = 200):
    return json(sys_json.loads(JSONEncoder.dumps(data)), status=status)


class _baseView(HTTPMethodView):
    """
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用
    views.POST  : --> query list
    
    
     __init_subclass__如果不调 super().__init_subclass__()，就会阻断 Generic 的设置流程
     python 3.12 的 typing 中，_generic_class_getitem（即 XXXView[Something]的 __class_getitem__）会访问 cls.__parameters__，
     如果 __init_subclass__链断了，__parameters__就不存在
     
    HTTPMethodView __init_subclass__阻断了 Generic的初始化链。导致 __parameters__没被写入
    """

    routePath: str = "/" 

    def response_json(self, data: Result | Page_Result, status: int = 200):
        return response_json(data, status=status)

    @staticmethod
    def is_integer(s: str | bytes | bytearray):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_Session(request: Request) -> Tuple[Session, SessionDict]:
        """
        获取 mem_session
        """
        sDict = request.ctx.Session
        session = request.app.ctx.extensions["Session"]
        return session, sDict

    @staticmethod
    def choose(request: Request, kyes: list | tuple, valueNone: bool = False):
        """
        选择dict 中 指定的key
        @param request 请求参数
        @param kyes 指定的key
        @param valueNone 当request.json key不存在时,返回{ ... ,key:None,...}
        """
        data = request.json
        return utils.choose(data, kyes, valueNone)

    def createContentDisposition(self, fileName):
        """
        创建内容描述header
        Content-Disposition:"attachment;filename*=UTF-8....."
        """
        encoded_filename = quote(fileName, encoding="utf-8")
        content_disposition = f"attachment;filename*=UTF-8''{encoded_filename}"
        headers = {
            "Content-Disposition": content_disposition,
            "Access-Control-Expose-Headers": "Content-Disposition",
        }
        return headers

    async def zip_directory(self, folder_path, output_zip):
        """
        将给定的文件夹压缩成ZIP文件。

        :param folder_path: 要压缩的文件夹路径
        :param output_zip: 输出的ZIP文件路径
        """
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # 计算文件的完整路径
                    file_path = os.path.join(root, file)
                    # 在ZIP文件中保持原来的目录结构
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    def get_folder_size(self, folder_path):
        """Return the total size of a folder in bytes using pathlib."""
        total_size = 0
        for path in Path(folder_path).rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size

    async def getSize(self, filePath: str):
        size = (
            os.path.getsize(filePath)
            if os.path.isfile(filePath)
            else self.get_folder_size(filePath)
        )
        return size

    async def response_head(self, size: int, fileName: str = None):
        """
        返回响应头信息
        @param size 文件大小
        @param fileName 文件名
        """
        response = json({})
        response.headers.update(
            {
                "Accept-Ranges": "bytes",
                "Content-Length": size,
                "Content-Type": "application/octet-stream",
            }
        )
        if fileName:
            headers = self.createContentDisposition(fileName)
            response.headers.update(headers)
        return response

    async def response_size(self, fullPath: str):
        """
        文件或目录大小
        如果是文件包括文件名
        """
        size = (
            os.path.getsize(fullPath)
            if os.path.isfile(fullPath)
            else self.get_folder_size(fullPath)
        )
        fileName = None
        if os.path.isfile(fullPath):
            fileName = os.path.basename(fullPath)
        return await self.response_head(size, fileName)

    @staticmethod
    def parseRange(request: Request, *, filePath: str = None, fileSize: int = None):
        """
        解析 HTTP.HEADER.Range 参数
        @param request 请求参数
        @param filePath 文件路径
        @param fileSize 文件大小
        return (start, end, fileSize)

        """
        params = [param for param in (filePath, fileSize) if param is not None]
        if len(params) > 1:
            raise ValueError("Exactly one of filePath or fileSize must be provided.")

        if fileSize is None:
            fileSize = os.path.getsize(filePath)
        range_header = request.headers.get("Range")
        if range_header:
            unit, ranges = range_header.split("=")
            if unit != "bytes":
                raise Exception("Only byte ranges are supported")

            start, end = map(lambda x: int(x) if x else None, ranges.split("-"))
            if start is None:
                start = fileSize - end
                end = fileSize - 1
            elif end is None or end >= fileSize:
                end = fileSize - 1
        return start, end, fileSize

    async def get_file_partial(self, request: Request, filePath: str):
        if os.path.isfile(filePath):
            fileName = os.path.basename(filePath)
        headers = self.createContentDisposition(fileName)
        file_size = os.path.getsize(filePath)
        headers.update(
            {
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }
        )
        start, end, _ = self.parseRange(request, fileSize=file_size)
        # return await file_stream(filePath,status=206, headers=headers )  # 未执行完 finally 就开始执行
        # return await file(filePath, headers=headers)  # 使用 file 适用于较小的文件 传送完整文件
        # return await file(filePath,filename= fileName)  # 使用 file 适用于较小的文件 中文名乱码
        # 返回二进制数据
        # 读取文件内容为二进制数据
        data: bytes = None
        with open(filePath, "rb") as f:
            f.seek(start)
            size = end - start + 1
            data = f.read(size)
        return raw(data, status=206, headers=headers)

    @staticmethod
    def usable_args(request: Request) -> dict:
        """
        去除列表
        request.args={name:['123'],groups:["a","b"]}
        return {name:'123',groups:["a","b"]}
        """
        args: dict = request.args
        data_result = {}
        for key in args:
            value = args.get(key)
            if len(value) == 1:
                data_result.update({key: value[0]})
            else:
                data_result.update({key: value})
        return data_result

    @staticmethod
    async def save_body(request: Request, root: str):
        # 保存上传的内容
        subDir = getDateFolder(format="%Y-%m-%d-%H-%M-%S")
        filePath = os.path.join(root, getDateFolder(), f"{subDir}.data")
        filePath = os.path.abspath(filePath)  # 转换为 os 所在系统路径
        folder = os.path.dirname(filePath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        async with aiofiles.open(filePath, "wb") as f:
            await f.write(request.body)
        # end 保存上传的内容

    @staticmethod
    async def parser_multipart_body(
        request: Request,
    ) -> Tuple[Dict[str, tuple | Any], Dict[str, multipart.MultipartPart]]:
        """
        解析内容: multipart/form-data; boundary=------------------------XXXXX,
        的内容
        """
        env = {
            "REQUEST_METHOD": "POST",
            "CONTENT_LENGTH": request.headers.get("content-length"),
            "CONTENT_TYPE": request.headers.get("content-type"),
            "wsgi.input": BytesIO(request.body),
        }
        data, file = multipart.parse_form_data(env)
        data_result = {}
        # log.info(data.__dict__)
        for key in data.__dict__.get("dict"):
            value = data.__dict__.get("dict").get(key)
            if len(value) == 1:
                data_result.update({key: value[0]})
            else:
                data_result.update({key: value})
        # log.info(data_result)
        return data_result, file

    async def save_file(self, file: File, path: str):
        """
        保存上传的文件
        file.name
        """
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if os.path.exists(path):
            raise Exception("{} Exists".format(path))
        async with aiofiles.open(path, "wb") as f:
            await f.write(file.body)

    async def _save_file(
        self, request: Request, *savePath: str, fileFieldName: str = None
    ):
        """
        保存上传的文件
        """
        p_len = len(savePath)
        if fileFieldName is not None and p_len == 1:
            file = request.files.get(fileFieldName)
            await self.save_file(file, *savePath)
        elif p_len == len(request.files):
            i: int = 0
            for file in request.files:
                file = request.files.get("file")
                await self.save_file(file, savePath[i])
                i += 1

    def getFullPath(self, root, fileName: str) -> Tuple[str, str]:
        """
        获取去路径和相对路径
        """
        filePath = "/".join(["", getDateFolder(), fileName])
        fullPath = os.path.join(root, filePath[1:])

        return fullPath, filePath


class BaseClsView(_baseView):
    def __init__(self, request: Request, *args, **kwargs) -> None:
        self.args=args
        self.kwargs=kwargs
        self._request = request

    @property
    def request(self):
        """
        请求对象
        """
        return self._request

    @property
    def match_info(self):
        """url 模板参数"""
        return self.request.match_info

    @property
    def json(self):
        """json 参数"""
        return self.request.json

    @property
    def config(self):
        """
        应用配置
        """
        return self.request.app.config

    @property
    def app(self):
        """
        应用对象
        """
        return self.request.app

    """
    async def get(self):
        raise NotImplementedError(f"get '{self.routePath}' not implemented")

    async def post(self):
        raise NotImplementedError(f"post '{self.routePath}' not implemented")

    async def put(self):
        raise NotImplementedError(f"put '{self.routePath}' not implemented")

    async def delete(self):
        raise NotImplementedError(f"delete '{self.routePath}' not implemented")
    """

    @classmethod
    def as_view(cls, *class_args: any, **class_kwargs: any):
        """
        将类转换为视图对象
        class_args: 类的参数  现在不支持
        class_kwargs: 类的参数参数字典  现在不支持
        """

        def view(*args, **kwargs):
            self = view.view_class(*args, **kwargs)
            return self()

        if cls.decorators:
            view.__module__ = cls.__module__
            for decorator in cls.decorators:
                view = decorator(view)

        view.view_class = cls  # type: ignore
        view.__doc__ = cls.__doc__
        view.__module__ = cls.__module__
        view.__name__ = cls.__name__
        return view
    async def __call__(self) -> BaseHTTPResponse:
        try:
            method = self.request.method
            handler = getattr(self, method.lower(), None)
            if handler is None:
                raise NotImplementedError(f"Method {method} not implemented")
            return await handler()
        except Exception as e:
            log.err("执行方法异常",e) 
            return self.response_json(e, status=500) 


@deprecated("该类已废弃，请使用 BaseClsView 类替代")
class BaseView(_baseView):
    """
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用
    views.POST  : --> query list
    views.PUT   :---> Add
    view.PATCH    :---> Edit
    view.DELETE :---> del
    """

    """
    类属性：
    路由使用路径
    """
    pass
