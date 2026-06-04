
from sanic import Request
from sanic.response import file
from . import resource_baseView


class Res_Image_View(resource_baseView):
    routePath = "/img/<pk:int>" 
    def pk(self):
        return self.match_info.get("pk")
    async def get(self ):
        """
        显示图片
        """
        fullPath = await self.getLocalPathById(self.pk())
        return await file(fullPath, mime_type="image/jpeg")


class Res_Video_View(resource_baseView):
    routePath = "/video/<pk:int>"
    def pk(self):
        return self.match_info.get("pk")
    async def get(self ):
        """
        显示视频文件
        """
        fullPath = await self.getLocalPathById(self.pk())
        return await file(fullPath, mime_type="image/jpeg")


class Res_thumbnail_View(resource_baseView):
    routePath = "/img/thumbnail/<pk:int>/<w:int>/<h:int>"
    def __init__(self, request: Request, pk: int, w: int = 208, h: int = 117, *args, **kwargs) -> None:
        self.pk = pk
        self.w = w
        self.h = h
        super().__init__(request, *args, **kwargs)
    async def get(self):
        """
        略缩图
        """
        fullPath = await self.getLocalPathById(self.pk)
        return await self.screenshot_image(fullPath, self.w, self.h)


class Res_Poster_View(resource_baseView):
    routePath = "/video/poster/<pk:int>/<w:int>/<h:int>"
    def __init__(self, request: Request, pk: int, w: int = 208, h: int = 117, *args, **kwargs) -> None:
        self.pk = pk
        self.w = w
        self.h = h
        super().__init__(request, *args, **kwargs)

    async def get(self ):
        """
        视频截图
        视频第一帧作为 poster
        未使用可能需要
        """
        fullPath = await self.getLocalPathById(self.pk)
        return await self.screenshot(fullPath, self.w, self.h)
