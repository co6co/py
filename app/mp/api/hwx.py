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
from sqlalchemy.sql import Select
from co6co_sanic_ext.model.res.result import Result
from view_model.biz.upload_view import syncCheckEntity,createResourceUUID,saveResourceToDb,alarm_success
  

hwx_api = Blueprint( "hwx", url_prefix="/nyzh/pubApi")  
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
            return text("200-ok")
    except Exception as e:
        log.err(f"上告HWX警失败：{e}")  
        res:m.Response=m.Response.fail( message=e)
        return JSON_util.response(res)  

@hwx_api.route("/ipncAlive",methods=["POST",])
async def post(  request:Request):
    """
    心跳
    """
    return text("200-ok")