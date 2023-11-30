from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic_ext import Extend
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util,json

from view_model.base_view import  BaseMethodView
from model.pos.right import bizDevicePo,bizResourcePO
from model.enum import resource_category,resource_image_sub_category
from typing import List,Optional ,Dict,Any
import view_model.biz.upload_ as m
from co6co.utils import log,getDateFolder
import os
from model.pos.right import bizResourcePO  ,bizAlarmAttachPO
import multipart  ,io

def get_upload_path(appConfig)->str|None:
    """
    获取上传路径
    """ 
    if "biz_config" in appConfig and "upload_path" in appConfig.get("biz_config"):
        root=appConfig.get("biz_config").get("upload_path")
        return root
    log.warn("未配置上传路径：请在配置中配置[biz_config-->upload_path]")
    return None

#@lru_cache(maxsize=20)
async def get_Device_id( db:DbOperations,param:m.Box_base_Param,upgrade=False): 
    one=await db.get_one(bizDevicePo.id,bizDevicePo.boardId==param.BoardId) 
    if one==None:
        po=bizDevicePo()
        po.boardId=param.BoardId
        po.innerIp=param.BoardIp
        po.ip=param.ip
        db.add(po)
        one=await db.get_one(bizDevicePo.id,bizDevicePo.boardId==param.BoardId) 
        return one
    elif upgrade: 
        po=await db.get_one_by_pk(bizDevicePo,one)  
        po.innerIp=param.BoardIp
        po.ip=param.ip 
    return one  
async def saveResourceToDb(opt:DbOperations,device_id, category:resource_category,  path:str,sub_category:int=None)->bizResourcePO:
    """
    资源保存
    """ 
    po=bizResourcePO()
    po.category= category.val
    po.subCategory=sub_category
    po.url=path

    po.deviceId=device_id 
    opt.add_all([po])
    return po
async def query_reource_id_by_fileName(db:DbOperations,fileName)->int:
    """
    通过文件名查询视频资源ID
    """ 
    one=await db.get_one(bizResourcePO.id,bizResourcePO.url.like(f"%{fileName}%")) 
    return one

        
class Video_Upload_View(BaseMethodView):
    """
    盒子上传视频，请求不需要认证
    """  
    async def post(self,request:Request): 
        """
        上传视频，
        现有客户端才有服务器,需配合客户端。
        """  
        p=m.Video_Param()
        data,files= await self.parser_multipart_body(request) 
        log.warn(request.headers)
        log.warn(data)
        p.__dict__.update(data)   
        p.ip=request.client_ip
        config=request.app.config
        root=get_upload_path(config)
        if root==None:return  JSON_util.response(m.Video_Response.fail("未配置上传目录"))
        # 1. 保存文件
        file:multipart.MultipartPart=files.get("Video") 
        if file==None:return  JSON_util.response(m.Video_Response.fail("上传[file['Video']]未找到视频文件")) 
        fullPath,filePath=self.getFullPath(root,file.filename) 
        file.save_as(fullPath)
        #2. 保存到数据库
        async with request.ctx.session as session:  
            session:AsyncSession=session
            operation=DbOperations(session)
            device_id=await get_Device_id(operation,p,True)
            po =await saveResourceToDb(operation,device_id,resource_category.video, filePath) 
            await session.commit() 
        # 返回响应 
        res:m.Video_Response=m.Video_Response.success() 
        log.err(f"返回的VideoID：{po.id}")
        res.VideoId=po.id
        request.files.get("Video")
        return JSON_util.response(res)


class Alarm_Upload_View(BaseMethodView):
    """
    盒子告警
    """
    async def _saveImage(self, request:Request,fileName:str, base64:str):
        config=request.app.config
        root=get_upload_path(config)
        if root==None:return  JSON_util.response(m.Video_Response.fail("未配置上传目录"))
        from co6co.utils.File import File
        fullPath,filePath=self.getFullPath(root,fileName) 
        File.writeBase64ToFile(fullPath,base64)
        return filePath 
    
    
    async def post(self, request:Request): 
        """
        上传告警信息
        """
        p=m.Alert_Param() 
        p.__dict__.update(request.json )
        p.ip=request.client_ip 
       
        #2. 保存到数据库
        async with request.ctx.session as session:  
            session:AsyncSession=session
            operation=DbOperations(session)
            device_id=await get_Device_id(operation,p)
            ## 2.1 保存图片 
            p1=await self._saveImage(request,f"{p.AlarmId}_src.jpeg",p.ImageData ) 
            p2=await self._saveImage(request,f"{p.AlarmId}_Labeled.jpeg",p.ImageDataLabeled ) 
            imagePo1:bizResourcePO=await saveResourceToDb(operation,device_id,resource_category.image,p1,sub_category=resource_image_sub_category.raw.val)
            imagePo2:bizResourcePO=await saveResourceToDb(operation,device_id,resource_category.image,p2,sub_category=resource_image_sub_category.marked.val)
            await operation.commit()
            po=p.to_po() 
            vid= await query_reource_id_by_fileName(operation,p.VideoFile) 
            po.videoId=vid
            po.rawImageId=imagePo1.id
            po.markedImageId=imagePo2.id 
            poa=bizAlarmAttachPO() 
            log.warn(type(p.Result))
            poa.result=json.dumps(p.Result) 
            poa.gps=json.dumps(p.GPS)
            poa.media=json.dumps(p.Media )
            session.add(po)
            log.warn(f"提交前：{po.id}")
            await operation.commit()
            log.warn(f"提交后：{po.id}")
            poa.id=po.id
            operation.add_all([poa]) 
            await operation.commit()
            
        res:m.Response=m.Response.success()
        return JSON_util.response(res)
    
