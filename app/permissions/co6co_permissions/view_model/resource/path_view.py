
from sanic import Request
from sanic.response import file
from . import resource_baseView


class Image_View(resource_baseView):
    routePath = "/img"

    async def get(self ):
        """
        显示图片
        """
        fullPath = await self.getLocalPath( )
        return await file(fullPath, mime_type="image/jpeg")


class Video_View(resource_baseView):
    routePath = "/video"

    async def get(self ):
        """
        显示视频文件
        """
        fullPath = await self.getLocalPath( )
        return await file(fullPath, mime_type="image/jpeg")


class thumbnail_View(resource_baseView):
    routePath = "/thumbnail/<w:int>/<h:int>"
    def __init__(self, request: Request, w: int = 208, h: int = 117, *args, **kwargs) -> None:
        self.w = w
        self.h = h
        super().__init__(request, *args, **kwargs)

    async def get(self ):
        """
        略缩图
        """
        fullPath = await self.getLocalPath( )
        return await self.screenshot_image(fullPath, self.w, self.h)


class Poster_View(resource_baseView):
    routePath = "/poster/<w:int>/<h:int>"
    def __init__(self, request: Request, w: int = 208, h: int = 117, *args, **kwargs) -> None:
        self.w = w
        self.h = h
        super().__init__(request, *args, **kwargs)

    async def get(self ):
        """
        视频截图
        视频第一帧作为 poster
        未使用可能需要
        """
        fullPath = await self.getLocalPath( )
        return await self.screenshot(fullPath, self.w, self.h)
