from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic_ext import Extend
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util

from view_model.base_view import  BaseMethodView
from model.pos.right import bizDevicePo,bizResourcePO
from model.enum import resource_category
from typing import List,Optional 
import view_model.biz.upload_ as m
from co6co.utils import log,getDateFolder
import os
from model.pos.right import bizResourcePO
from functools import lru_cache
import aiofiles


def get_upload_path(appConfig)->str|None:
    """
    获取上传路径
    """ 
    if "biz_config" in appConfig and "upload_path" in appConfig.get("biz_config"):
        root=appConfig.get("biz_config").get("upload_path")
        return root
    log.warn("未配置上传路径：请在配置中配置[biz_config-->upload_path]")
    return None
async def save_body(request:Request):
    root=get_upload_path(request.app.config)
    filePath=os.path.join(root,getDateFolder(),f"{getDateFolder(format="%Y-%m-%d") }.json")
    filePath=os.path.abspath(filePath) # 转换为 os 所在系统路径 
    folder=os.path.dirname(filePath) 
    
    log.warn(f"arg:{request.args}")
    log.warn(f"form:{request.form}")
    if not os.path.exists(folder):os.makedirs(folder)
    async with aiofiles.open(filePath, 'wb') as f:
        await f.write( request.body) 


class Video_Upload_View(BaseMethodView):
    """
    盒子上传视频，请求不需要认证
    """ 
    #@lru_cache(maxsize=20)
    async def get_Device_id(db:DbOperations,param:m.Video_Param): 
        one=await db.get_one(bizDevicePo.id,bizDevicePo.boardId==param.BoardId) 
        if one==None:
            po=bizDevicePo()
            po.boardId=param.BoardId
            po.innerIp=param.BoardIp
            po.ip=param.ip
            db.add(po)
            return po.id
        return one.get(bizDevicePo.id.key)

    async def post(self,request:Request): 
        """
        上传视频，
        现有客户端才有服务器,需配合客户端。
        """  
        p=m.Video_Param()
        await save_body(request)
        res:m.Video_Response=m.Video_Response.success() 
        res.VideoId=0
        request.files.get("Video")
        
        return JSON_util.response(res)
        p.__dict__.update(request.json)
        config=request.app.config
        root=get_upload_path(config)
        if root==None:return  JSON_util.response(m.Video_Response.fail("未配置上传目录"))
        # 1. 保存文件
        file=request.files.get("Video")
        if file==None:return  JSON_util.response(m.Video_Response.fail("上传[file['Video']]未找到视频文件")) 
        filePath=os.path.join("/",getDateFolder(),file.name)
        await self.save_file(file,os.path.join(root,filePath[1:])) 
        #2. 保存到数据库
        async with request.ctx.session as session:  
            session:AsyncSession=session
            operation=DbOperations(session)
            device_id=await self.get_Device_id(operation,p)
            po=bizResourcePO()
            po.category= resource_category.video.val
            po.url=filePath
            po.deviceId=device_id
            operation.add_all([po])
            await session.commit() 
        # 返回响应 
        res:m.Video_Response=m.Video_Response.success() 
        res.VideoId=po.id
        request.files.get("Video")
        return JSON_util.response(res)
    
class Alarm_Upload_View(BaseMethodView):
    """
    盒子告警
    """
    async def post(self, request:Request): 
        """
        上传告警信息
        """
        p=m.Alert_Param()
        p.AlarmId=getDateFolder(format="%Y-%m-%d") 
        await save_body(request)
        res:m.Response=m.Response.success()
        return JSON_util.response(res)
    
