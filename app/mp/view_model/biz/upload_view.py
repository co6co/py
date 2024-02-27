from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request ,Sanic
from sanic_ext import Extend
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util
import json

from view_model.base_view import  BaseMethodView
from model.pos.biz import bizBoxPO,bizResourcePO

from model.enum import resource_category,resource_image_sub_category,device_type
from typing import List,Optional ,Dict,Any
import view_model.biz.upload_ as m
from co6co.utils import log,getDateFolder
import os,io
from model.pos.biz import bizResourcePO  ,bizAlarmAttachPO,bizAlarmTypePO,bizAlarmPO

import multipart,uuid,datetime
from view_model import get_upload_path
from view_model.aop .alarm_aop import Alarm_Save_Succ_AOP
from utils import createUuid

 
#@lru_cache(maxsize=20)
async def get_Device_id( db:DbOperations,param:m.Box_base_Param,upgrade=False): 
    one=await db.get_one(bizBoxPO.id,bizBoxPO.uuid==param.BoardId) 
    if one==None: 
        po=bizBoxPO()
        po.uuid=param.BoardId
        po.innerIp=param.BoardIp
        po.ip=param.ip 
        db.add(po)
        await db.commit()  
        return po.id
    elif upgrade: 
        po=await db.get_one_by_pk(bizBoxPO,one)  
        po:bizBoxPO=po
        po.innerIp=param.BoardIp
        po.ip=param.ip 
        await db.commit() 
    return one 


def createResourceUUID( fileName:str=None)->str:
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
            if len(result)==36:return result
        elif len(name)==36:return fileName 
    return str(createUuid())   
    
async def saveResourceToDb(opt:DbOperations,device_id,category:resource_category,path:str,sub_category:int=None,uid:str=None)->bizResourcePO:
    """
    资源保存
    """ 
    if uid==None:uid=createResourceUUID()
    po=bizResourcePO()
    po.category= category.val
    po.subCategory=sub_category
    po.url=path 
    po.uid=uid
    po.boxId=device_id 
    opt.add_all([po])
    return po  

async def syncCheckEntity(app:Sanic,param:dict):
    """
    同步 检测类型 表 
    param: {Type:"告警类型","Description":"告警描述"}
    """ 
    if "Type" in  param and "Description" in  param:
        session:AsyncSession=app.ctx.session_fatcory()
        opt=DbOperations(session)
        try: 
            one:bizAlarmTypePO=await opt.get_one(bizAlarmTypePO,bizAlarmTypePO.alarmType==param.get("Type")) 
            if one==None:
                one=bizAlarmTypePO()
                one.alarmType=param.get("Type")
                one.desc=param.get("Description")
                opt.add(one)
            else:
                one.desc=param.get("Description")
                one.updateTime=datetime.datetime.now()
            await opt.commit()
        except Exception as e:
            log.err(f"执行同步syncCheckEntity，失败{e}")
        finally:
            await opt.db_session.close() 
    else: log.warn("同步告警类型参数不正确！")  


@Alarm_Save_Succ_AOP 
def alarm_success(request:Request ,po:bizAlarmPO):
    """
    apo
    """
    return None
   
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
        if file==None: 
            log.warn("未找到上传的视频内容Video")
            return JSON_util.response(m.Video_Response.fail("上传[file['Video']]未找到视频文件"))
        
        try:
            #2. 保存到数据库
            async with request.ctx.session as session:
                ## 1.1 查询以前是否上传过
                session:AsyncSession=session
                operation=DbOperations(session)
                uid=createResourceUUID(file.filename)
                
                isExist=await operation.exist(bizResourcePO.uid==uid,column=bizResourcePO.id)
                
                if not isExist: 
                    fullPath,filePath=self.getFullPath(root,file.filename)
                    file.save_as(fullPath) 
                    device_id=await get_Device_id(operation,p,True) 
                    po:bizResourcePO =await self.saveResourceToDb(operation,device_id,resource_category.video, filePath,uid= uid) 
                    await session.commit() 
                    log.succ(f"返回的VideoID：uid:{uid},id:{po.id}") 
                else:
                    log.warn(f"返回的VideoID：uid:{uid} 已存在！") 
                # 返回响应 
                res:m.Video_Response=m.Video_Response.success()  
                res.VideoId=uid 
                return JSON_util.response(res)
        except Exception as e:
            log.err(f"上传视频失败：{e}")
            # 为了继续上传影响业务 ，返回成功
            res:m.Video_Response=m.Video_Response.fail()  
            res.VideoId="-1" 
            return  JSON_util.response(res)
              


class Alarm_Upload_View(BaseMethodView):
    """
    盒子告警
    """ 
    async def _saveImage(self,request:Request,fileName:str, base64:str):
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
        log.warn(p.VideoFile)
        request.app.add_task( syncCheckEntity( request.app,p.Result),name="同步检测类型")
        ## debug
        await self.save_body(request,get_upload_path(request.app.config))
        try:
            #2. 保存到数据库
            async with request.ctx.session as session:  
                session:AsyncSession=session
                opt=DbOperations(session)
                device_id=await get_Device_id(opt,p)
                ## 2.1 保存图片 
                u1=createResourceUUID()
                u2=createResourceUUID()
                p1=await self._saveImage(request,f"SRC_{u1}.jpeg",p.ImageData ) 
                p2=await self._saveImage(request,f"Labeled_{u2}.jpeg",p.ImageDataLabeled ) 
                await saveResourceToDb(opt,device_id,resource_category.image,p1,sub_category=resource_image_sub_category.raw.val,uid= u1)
                await saveResourceToDb(opt,device_id,resource_category.image,p2,sub_category=resource_image_sub_category.marked.val,uid=u2)
                await opt.commit()
                po=p.to_po() 
                po.boxId=device_id
                result= await opt.exist(bizAlarmPO.uuid==po.uuid) 
                if result: 
                    res:m.Response=m.Response.success(message=f"数据“{po.uuid}”重复上传")
                    #通知公众号订阅者
                    alarm_success(request,po)
                    log.warn(f"告警信息uuid重复{po.uuid}")
                    return JSON_util.response(res)
                #关联视频资源 视频资源可能为空，有UUID 但没有资源
                if p.VideoFile !=None and p.VideoFile!='':po.videoUid=createResourceUUID(p.VideoFile)
                po.rawImageUid=u1
                po.markedImageUid=u2
                poa=bizAlarmAttachPO()  # 附加信息
                poa.result=json.dumps(p.Result) 
                poa.gps=json.dumps(p.GPS)
                poa.media=json.dumps(p.Media )
                po.alarmAttachPO=poa 
                session.add(po) 
                await opt.commit()  
                res:m.Response=m.Response.success()
                #通知公众号订阅者
                alarm_success(request,po)
                return JSON_util.response(res) 
        except Exception as e:
            log.err(f"上告警失败：{e}")
            res:m.Response=m.Response.fail( message=e)
            return JSON_util.response(res) 
