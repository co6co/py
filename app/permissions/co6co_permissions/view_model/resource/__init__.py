
from ...services.configCache import get_upload_path
from .utils import resize_image, screenshot, getTempFileName
from co6co .utils import log
from ..base_view import AuthMethodView
import json
import io
import os
from PIL import Image
from sanic import Request
from sanic.response import text, raw, empty, file, file_stream, ResponseStream
from ...model.pos.resource import resourcePO
from co6co_db_ext.db_utils import QueryOneCallable
from sqlalchemy.sql import Select


class resource_baseView(AuthMethodView):
    async def getRersourcePath(self, request: Request, path):
        uploadRoot = await get_upload_path(request)
        fullPath = os.path.join(uploadRoot, path[1:])
        return os.path.abspath(fullPath)

    async def getLocalPathById(self, request: Request, pk: int) -> str:
        call = QueryOneCallable(self.get_db_session(request))
        data = await call(Select(resourcePO.url).filter(resourcePO.id == pk), isPO=False)
        if data != None:
            return await self.getRersourcePath(request, data["url"])
        return None

    async def getLocalPath(self, request: Request) -> str:
        for k, v in request.query_args:
            if k == "path":
                path = v
        return await self.getRersourcePath(request, path)

    async def screenshot(self, fullPath: str, w: int = 208, h: int = 117, isFile: bool = True):
        """
        视频截图
        视频第一帧作为 poster
        """
        if os.path.exists(fullPath):
            tempPath = await screenshot(fullPath, w, h, isFile=True, useBytes=True)
            if tempPath == None:
                return empty(status=404)
            return raw(tempPath,  status=200, headers=None,  content_type="image/jpeg")
        return empty(status=404)

    async def screenshot_image(self, fullPath: str, w: int = 208, h: int = 117):
        """ 
        略缩图
        """
        if os.path.exists(fullPath):
            im = Image.open(fullPath)
            bytes = io.BytesIO()
            im.thumbnail((w, h))
            im.save(bytes, "PNG")
            return raw(bytes.getvalue(),  status=200, headers=None,  content_type="image/jpeg")
        return empty(status=404)
