
from ...services.configCache import get_upload_path
from .utils import resize_image, screenshot, getTempFileName
from co6co .utils import log
from ..base_view import AuthMethodView 
import io
import os
from PIL import Image
from io import BytesIO
import requests
from sanic import Request
from sanic.response import text, raw, empty, file, file_stream, ResponseStream
from ...model.pos.resource import resourcePO

from sqlalchemy.sql import Select
from sanic.request.form import File
from co6co.utils import hash, getDateFolder
from ...model.enum import resource_category
from datetime import datetime
import uuid
from co6co.data.result import Result


class FileResult:
    fullPath: str
    path: str
    hash: str
    name: str
    size: int

    def toPo(self, category: resource_category, subCategory: int = 0):
        po = resourcePO()
        po .uid = uuid.uuid4()
        po .category = category.val
        po .subCategory = subCategory
        po .hash = self.hash
        po .url = self.path
        po .metaName = self.name
        po .size = self.size
        po.createTime = datetime.now()
        return po


class resource_baseView(AuthMethodView):
    async def getRersourcePath(self,   path):
        """
        获取资源路径

        @param request   获取上传根目录
        @param path  eg. /upload/2022/01/01/123.jpg

        return fullPath
        """
        uploadRoot = await get_upload_path(self.request)
        fullPath = os.path.join(uploadRoot, path[1:])
        return os.path.abspath(fullPath)

    async def getLocalPathById(self,  pk: int) -> str:
        """
        通过 id 获取本地路径
        return fullPath
        """ 
        select=Select(resourcePO.url).filter(resourcePO.id == pk)
        data = await self.actuator.query_one_mappings(select)
     
        if data is not None:
            return await self.getRersourcePath( data["url"])
        return None

    async def getLocalPath(self ) -> str:
        path = ""
        for k, v in self.request.query_args:
            if k == "path":
                path = v
        if path.startswith("http"):
            return path
        return await self.getRersourcePath(path)

    async def screenshot(self, fullPath: str, w: int = 208, h: int = 117, isFile: bool = True):
        """
        视频截图
        视频第一帧作为 poster
        """
        if fullPath.startswith('http') or os.path.exists(fullPath):
            isFile = not fullPath.startswith('http')
            tempPath = await screenshot(fullPath, w, h, isFile=isFile, useBytes=True)
            if tempPath is None:
                return empty(status=404)
            return raw(tempPath,  status=200, headers=None,  content_type="image/jpeg")
        return empty(status=404)

    async def readHttpImage(self, url):
        """
        从 url 中获取 图片
        """
        data = requests.get(url)
        if data.status_code == 200:
            data = BytesIO(data.content)
            im = Image.open(data)
            return im
        return None

    async def readLocalImage(path):
        if os.path.exists(path):
            im = Image.open(path)
            return im
        return None

    async def response_local_file(self,  fullPath: str):
        """
        响应本地文件
        """
        filePath = await self.getRersourcePath( fullPath)
        if os.path.exists(filePath):
            return await file(filePath)

    async def screenshot_image(self, fullPath: str, w: int = 208, h: int = 117):
        """ 
        @param fullPath 完整路径 url 或者本地路径
        略缩图
        """
        try:
            im = None
            if fullPath.startswith('http'):
                im = await self.readHttpImage(fullPath)
            elif os.path.exists(fullPath):
                im = await self.readLocalImage(fullPath)
            if im is not None:
                bytes = io.BytesIO()
                im.thumbnail((w, h))
                im.save(bytes, "PNG")
                return raw(bytes.getvalue(),  status=200, headers=None,  content_type="image/jpeg")
            return empty(status=404)
        finally:
            if im is not None:
                im.close()

    async def saveFile(self ):
        """
        保存文件到本地
        @param request  request.files.get("file") 
        return FileResult
        """
        try:
            fullPath = None
            basePath = await get_upload_path(self.request)
            if "file" not in self.request.files:
                return Result.fail("No file part in the request")
            file: File = self.request.files.get("file")

            size = len(file.body)  # await self. readFileSize(file)
            _, file_extension = os.path.splitext(file.name)
            fileName = getDateFolder(format='%Y-%m-%d-%H-%M-%S')
            fullPath, path = self.getFullPath(basePath, "{}{}".format(fileName, file_extension))
            await self.save_file(file, fullPath)
            result = FileResult()
            result.name = file.name
            result.hash = hash.file_sha256(fullPath)
            result.size = size
            result.fullPath = fullPath
            result.path = path
            return result
        except Exception as e:
            log.warn("保存文件失败：", fullPath, e)
            return  Result.fail(message=str(e))
