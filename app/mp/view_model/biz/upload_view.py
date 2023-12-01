from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request ,Sanic
from sanic_ext import Extend
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util,json

from view_model.base_view import  BaseMethodView
from model.pos.right import bizDevicePo,bizResourcePO

from model.enum import resource_category,resource_image_sub_category
from typing import List,Optional ,Dict,Any
import view_model.biz.upload_ as m
from co6co.utils import log,getDateFolder
import os,io
from model.pos.right import bizResourcePO  ,bizAlarmAttachPO,bizAlarmType
import multipart,uuid,datetime
from view_model import get_upload_path



#@lru_cache(maxsize=20)
async def get_Device_id( db:DbOperations,param:m.Box_base_Param,upgrade=False): 
    one=await db.get_one(bizDevicePo.id,bizDevicePo.boardId==param.BoardId) 
    if one==None:
        po=bizDevicePo()
        po.boardId=param.BoardId
        po.innerIp=param.BoardIp
        po.ip=param.ip
        db.add(po)
        await db.commit()  
        return po.id
    elif upgrade: 
        po=await db.get_one_by_pk(bizDevicePo,one)  
        po.innerIp=param.BoardIp
        po.ip=param.ip
        await db.commit() 
    return one  

 

class Upload_view(BaseMethodView): 
    def createResourceUUID(self, fileName:str=None)->str:
        """
        创建/或读取 资源UUID
        如果 file =="VIDEO_4AA8CBEB-59C7-4644-AD95-CCBC865E0FFD.mp4",从文件名获取UUID 
        返回: 36小写
        """ 
        if fileName!=None: 
            name,_=os.path.splitext(fileName)
            if "_" in name: 
                index=name.index("_")+1
                result=name[index:].lower()
                if result==36:return result
        return str(uuid.uuid1())   
    async def saveResourceToDb(self,opt:DbOperations,device_id,category:resource_category,path:str,sub_category:int=None,uid:str=None)->bizResourcePO:
        """
        资源保存
        """ 
        if uid==None:uid=self.createResourceUUID()
        po=bizResourcePO()
        po.category= category.val
        po.subCategory=sub_category
        po.url=path 
        po.uid=uid
        po.deviceId=device_id 
        opt.add_all([po])
        return po    
class Video_Upload_View(Upload_view):

    
    """
    盒子上传视频，请求不需要认证
    """  
    async def post(self,request:Request): 
        """
        上传视频，
        现有客户端才有服务器,需配合客户端。
        """  
        p=m.Video_Param()
        p.ip=request.client_ip
        #p.ip=self.forwarded['for'] 
        data,files= await self.parser_multipart_body(request)
        ## todo debug 
        await self.save_body(request,get_upload_path(request.app.config))
        p.__dict__.update(data)   
       
        config=request.app.config
        root=get_upload_path(config)
        if root==None:return  JSON_util.response(m.Video_Response.fail("未配置上传目录"))
        # 1. 保存文件
        file:multipart.MultipartPart=files.get("Video") 
        if file==None:return  JSON_util.response(m.Video_Response.fail("上传[file['Video']]未找到视频文件"))
        
        #2. 保存到数据库
        async with request.ctx.session as session:
            ## 1.1 查询以前是否上传过
            session:AsyncSession=session
            operation=DbOperations(session)
            uid=self.createResourceUUID(file.filename)
            isExist=await operation.exist(bizResourcePO.uid==uid,bizAlarmAttachPO.id)
            if not isExist: 
                fullPath,filePath=self.getFullPath(root,file.filename)
                file.save_as(fullPath) 
                device_id=await get_Device_id(operation,p,True) 
                await self.saveResourceToDb(operation,device_id,resource_category.video, filePath,uid= uid) 
                await session.commit() 
            else:
                session.close
        # 返回响应 
        res:m.Video_Response=m.Video_Response.success() 
        log.err(f"返回的VideoID：{uid}")
        res.VideoId=uid 
        return JSON_util.response(res)


class Alarm_Upload_View(Upload_view):
    """
    盒子告警
    """
    async def syncCheckEntity(self,app:Sanic,param:m.Alert_Param):
        """
        同步 检测类型 表
        """

        if "Type" in  param.Result and "Description" in  param.Result:
            session:AsyncSession=app.ctx.session_fatcory()
            opt=DbOperations(session)
            try: 
                one:bizAlarmType=await opt.get_one(bizAlarmType,bizAlarmType.type==param.Result.get("Type")) 
                if one==None:
                    one=bizAlarmType()
                    one.type=param.Result.get("Type")
                    one.desc=param.Result.get("Description")
                    opt.add(one)
                else:
                    one.desc=param.Result.get("Description")
                    one.updateTime=datetime.datetime.now()
                opt.commit()
            except Exception as e:
                log.err(f"执行同步syncCheckEntity，失败{e}")
            finally:
                await opt.db_session.close() 

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
        request.app.add_task(self. syncCheckEntity( request.app,p),name="同步检测类型")
        ## debug
        await self.save_body(request,get_upload_path(request.app.config))
        try:
            #2. 保存到数据库
            async with request.ctx.session as session:  
                session:AsyncSession=session
                operation=DbOperations(session)
                device_id=await get_Device_id(operation,p)
                ## 2.1 保存图片 
                u1=self.createResourceUUID()
                u2=self.createResourceUUID()
                p1=await self._saveImage(request,f"SRC_{u1}.jpeg",p.ImageData ) 
                p2=await self._saveImage(request,f"Labeled_{u2}.jpeg",p.ImageDataLabeled ) 
                await self.saveResourceToDb(operation,device_id,resource_category.image,p1,sub_category=resource_image_sub_category.raw.val,uid= u1)
                await self. saveResourceToDb(operation,device_id,resource_category.image,p2,sub_category=resource_image_sub_category.marked.val,uid=u2)
                await operation.commit()
                po=p.to_po()  
                po.videoUid=self.createResourceUUID(p.VideoFile)
                po.rawImageUid=u1
                po.markedImageUid=u2
                poa=bizAlarmAttachPO()  
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

        except Exception as e:
            res:m.Response=m.Response.fail(message=str(e))
            return JSON_util.response(res) 
