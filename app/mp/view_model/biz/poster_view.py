
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw,empty,file,file_stream
from co6co_sanic_ext.utils import JSON_util
import json
from model.enum import resource_category

from view_model import get_upload_path
from view_model.base_view import BaseMethodView,  AuthMethodView
from model.pos.biz import bizResourcePO
import os, cv2 , datetime  
from co6co .utils import log

from PIL import Image


class Image_View(BaseMethodView):
      
    # 按指定图像大小调整尺寸
    def resize_image(self, imreadImage, height = 208, width = 117):
        top, bottom, left, right = (0,0,0,0) 
        # 获取图片尺寸
        h, w, _ = imreadImage.shape
        
        # 对于长宽不等的图片，找到最长的一边
        longest_edge = max(h,w)
        
        # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。
        
        # RGB颜色
        BLACK = [0,0,0]
        # 给图片增加padding，使图片长、宽相等
        # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
        constant = cv2.copyMakeBorder(imreadImage, top, bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
        # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
        return cv2.resize(constant, (height, width))
     
    async def screenshot(self,fullPath:str,w:int=208,h:int=117,isFile:bool=True):
        """
        视频截图
        视频第一帧作为 poster
        """
        log.warn(fullPath)
        if (isFile and os.path.exists(fullPath)) or fullPath:
            try:
                
               
                cap =cv2.VideoCapture(fullPath) # 打开视频
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                #cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC , 5)
                ret,fram=cap.read()
                s=None
                if ret: 
                    if not os.path.exists("tmp"):os.makedirs("tmp")
                    s='tmp/frame_%s.jpg' %datetime.datetime.now().strftime('%H%M%S%f')
                    fram=self.resize_image(fram,w,h)
                    cv2.imwrite(s, fram) 
                    res= await file(fram,mime_type="image/jpeg")  
                    return res  
            finally:
                #if s !=None:os.remove(s)
                cap.release()
        return empty(status=404) 
    
    async def screenshot_image(self,fullPath:str,w:int=208,h:int=117):
        """ 
        略缩图
        """ 
        if os.path.exists(fullPath):
            try:  
                if not os.path.exists("tmp"):os.makedirs("tmp")
                s='tmp/frame_%s.jpg' %datetime.datetime.now().strftime('%H%M%S%f') 
            
                im = Image.open (fullPath)
                im.thumbnail((200,100))
                im.save(s,'PNG') 
                '''
                截图来 上下有很多黑边
                fram=cv2.imread(fullPath)
                fram=self.resize_image(fram,w,h)  
                cv2.imwrite(s, fram) 
                '''
                res= await file(s,mime_type="image/jpeg")  
                return res  
            finally:
                if s !=None:os.remove(s) 
        return empty(status=404)  
    async def get(self, request:Request ,uid:str,w:int,h:int):
        """
        视频截图 视频第一帧作为 poster
        图片 略缩图
        """  
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session)
            while(True): 
                dic= await operation.get_one([bizResourcePO.url,bizResourcePO.category],bizResourcePO.uid==uid) 
                if dic==None:break
                else: 
                    type:int=dic.get(bizResourcePO.category.key)
                    url=dic.get(bizResourcePO.url.key)
                    upload=get_upload_path(request.app.config)
                    fullPath=os.path.join(upload,url[1:]) 
                    log.warn(type)
                    if resource_category.video.val== type: 
                        return await self.screenshot(fullPath,w,h)
                    else:
                        return await self.screenshot_image(fullPath,w,h) 
        return empty(status=404)


class Poster_View(Image_View):   
    
    async def poster(self,request:Request ):
        """
        视频截图
        视频第一帧作为 poster
        未使用可能需要
        """
        for k,v in request.query_args:
            if k=="path": path=v 
        upload=get_upload_path(request.app.config)
        fullPath=os.path.join(upload,path[1:]) 
        return await self.screenshot(fullPath)
 
    
    async def get(self, request:Request ,uid:str):
        """
        视频截图
        视频第一帧作为 poster 
        """  
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session)
            while(True): 
                dic= await operation.get_one([bizResourcePO.url,bizResourcePO.category],bizResourcePO.uid==uid) 
                if dic==None:break
                else:
                   
                    type:int=dic.get(bizResourcePO.category.key)
                    url=dic.get(bizResourcePO.url.key)
                    upload=get_upload_path(request.app.config)
                    fullPath=os.path.join(upload,url[1:]) 
                    if resource_category.video.val== type:
                        log.info(request.args)
                        return await self.screenshot(fullPath,208,117)
                    else:
                        return await self.screenshot_image(fullPath) 
        return empty(status=404)