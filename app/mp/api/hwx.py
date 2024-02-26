from sanic import Blueprint
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request ,Sanic
from sanic_ext import Extend
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util
import json
from utils import createUuid 
from view_model.base_view import  BaseMethodView  
from model.enum import hwx_alarm_type, resource_category,resource_image_sub_category,device_type
from typing import List,Optional ,Dict,Any
import view_model.biz.upload_ as m
from co6co.utils import log,getDateFolder
import os,io
from model.pos.biz import bizResourcePO  ,bizAlarmAttachPO,bizAlarmTypePO,bizAlarmPO,bizBoxPO

import multipart,uuid,datetime
from view_model import get_upload_path
from view_model.aop .alarm_aop import Alarm_Save_Succ_AOP
from view_model.base_view import  BaseMethodView
from view_model.biz.upload_view import Upload_view
from sqlalchemy.sql import Select
from co6co_sanic_ext.model.res.result import Result
  

hwx_api = Blueprint( "hwx", url_prefix="/nyzh/pubApi") 



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
@staticmethod
def alarm_success(request:Request ,po:bizAlarmPO):
    return None
async def getSiteId(p:m.HWX_Param):
        """
        获取站点ID
        """
        try:
            snvr=p.snvr  
            if(len(snvr)==19):
                siteId= int(snvr[11:14]) #"SNVR-SZQA-Y007-0000"
                return siteId
            return None
        except Exception as e:
            log.err(f"获取Site Id失败,by {p.snvr}")
            return None 

@hwx_api.route("/alarmEvent",methods=["POST",])
async def alarmEvent(request:Request): 
    """
    上传告警信息
    """
    p=m.HWX_Param() 
    p.__dict__.update(request.json )
    
    try:
        #2. 保存到数据库
        session:AsyncSession=request.ctx.session
        type=p.break_rules
        alarmType=hwx_alarm_type.get(type)
        alarm={"Type":type,"Description":f"{type}描述"}
        if alarmType!=None:alarm.update({"Description":alarmType.label})
        log.warn(alarm)
        await  syncCheckEntity( request.app,alarm) 
        async with  session,session.begin():    
            ## 2.0 查询设备
            opt=DbOperations(session)  
            siteId= await getSiteId(p)
            select=( Select(bizBoxPO.id) .filter(bizBoxPO.siteId==siteId))
            exec=await session.execute(select)
            boxPO:bizBoxPO=exec.fetchone() 
            if(boxPO==None): return JSON_util.response( Result.fail(message=f"未找到站点关联的设备,siteId:{siteId}"))  
            boxId=boxPO.id
            log.succ(boxId)
            # 2.1  保存资源ID  
            resources=p.getResources()

            f=0
            vUUID=None
            Image1UUID=None 
            Image2UUID=None 
            for r in resources: 
                category=resource_category.image
                u1=createResourceUUID() 
                if (f==0):
                    vUUID=u1
                    category=resource_category.video
                elif (f==1):
                    Image1UUID=u1
                elif (f==2):
                    Image1UUID=u1
                else:
                    log.warn("上传图片数量超过保存数量")
                select=(Select(bizResourcePO.id).filter(bizResourcePO.url==r))
                exec=await session.execute(select)
                exist=exec.fetchone()
                if (exist==None): await saveResourceToDb(opt,boxId,category,r,uid= u1)
                else :log.warn(f"资源{r} 已存在:{exist}")
                f+=1
            po=p.to_po()  
            po.boxId=boxId 
            #关联有UUID 
            po.videoUid=vUUID
            po.rawImageUid=Image1UUID
            po.markedImageUid=Image2UUID 
            session.add(po) 
            await opt.commit()  
            res:m.Response=m.Response.success()
            #通知公众号订阅者
            alarm_success(request,po)
            return JSON_util.response(res) 
    except Exception as e:
        log.err(f"上告HWX警失败：{e}") 
        raise
        res:m.Response=m.Response.fail( message=e)
        return JSON_util.response(res)  

@hwx_api.route("/ipncAlive",methods=["POST",])
async def post(  request:Request):
    """
    心跳
    """
    return text("200-ok")