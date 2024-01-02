
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import Request
from sanic.response import text, raw, empty, file, file_stream, ResponseStream
from co6co_sanic_ext.utils import JSON_util
import json,io
from model.enum import resource_category

from view_model import get_upload_path
from view_model.base_view import BaseMethodView,  AuthMethodView
from model.pos.biz import bizResourcePO
import os 
from co6co .utils import log
from utils .cvUtils import resize_image, screenshot, getTempFileName 
from PIL import Image


class Image_View(BaseMethodView):
    async def screenshot(self, fullPath: str, w: int = 208, h: int = 117, isFile: bool = True):
        """
        视频截图
        视频第一帧作为 poster
        """  
        tempPath = await screenshot(fullPath, w, h, isFile=True,useBytes=True)
        if tempPath == None: return empty(status=404)
        return raw(  tempPath,  status=200, headers=None,  content_type="image/jpeg" )  

    async def screenshot_image(self, fullPath: str, w: int = 208, h: int = 117):
        """ 
        略缩图
        """
        if os.path.exists(fullPath):  
            im = Image.open(fullPath)
            bytes=io.BytesIO() 
            im.thumbnail((w, h)) 
            im.save(bytes,"PNG") 
            #im.save('s', 'PNG')
            return raw( bytes.getvalue(),  status=200, headers=None,  content_type="image/jpeg" ) 
            #  s = getTempFileName()
            #im.save(s, 'PNG')
            #return await file(s, mime_type="image/jpeg") 
        
            #if s != None:  os.remove(s)
        return empty(status=404)

    async def get(self, request: Request, uid: str, w: int, h: int):
        """
        视频截图 视频第一帧作为 poster
        图片 略缩图
        """
        async with request.ctx.session as session:
            session: AsyncSession = session
            operation = DbOperations(session)
            while (True):
                dic = await operation.get_one([bizResourcePO.url, bizResourcePO.category], bizResourcePO.uid == uid)
                if dic == None:
                    break
                else:
                    type: int = dic.get(bizResourcePO.category.key)
                    url = dic.get(bizResourcePO.url.key)
                    upload = get_upload_path(request.app.config)
                    fullPath = os.path.join(upload, url[1:])
                    log.warn(type)
                    if resource_category.video.val == type:
                        return await self.screenshot(fullPath, w, h)
                    else:
                        return await self.screenshot_image(fullPath, w, h)
        return empty(status=404)


class Poster_View(Image_View):

    async def poster(self, request: Request):
        """
        视频截图
        视频第一帧作为 poster
        未使用可能需要
        """
        for k, v in request.query_args:
            if k == "path":
                path = v
        upload = get_upload_path(request.app.config)
        fullPath = os.path.join(upload, path[1:])
        return await self.screenshot(fullPath)

    async def get(self, request: Request, uid: str):
        """
        视频截图
        视频第一帧作为 poster 
        """
        async with request.ctx.session as session:
            session: AsyncSession = session
            operation = DbOperations(session)
            while (True):
                dic = await operation.get_one([bizResourcePO.url, bizResourcePO.category], bizResourcePO.uid == uid)
                if dic == None:
                    break
                else:
                    type: int = dic.get(bizResourcePO.category.key)
                    url = dic.get(bizResourcePO.url.key)
                    upload = get_upload_path(request.app.config)
                    fullPath = os.path.join(upload, url[1:])
                    if resource_category.video.val == type:
                        log.info(request.args)
                        return await self.screenshot(fullPath, 208, 117)
                    else:
                        return await self.screenshot_image(fullPath)
        return empty(status=404)
