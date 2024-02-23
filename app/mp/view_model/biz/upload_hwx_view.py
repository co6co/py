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
from view_model.biz.upload_view import Upload_view
 
class Alarm_Upload_View(Upload_view): 
    @Alarm_Save_Succ_AOP
    @staticmethod
    def alarm_success(request:Request ,po:bizAlarmPO):
        return None
    
    async def post(self, request:Request): 
        """
        上传告警信息
        """
        p=m.HWX_Param() 
        p.__dict__.update(request.json )
        p.ip=request.client_ip
        ## debug
        await self.save_body(request,get_upload_path(request.app.config))
        try:
            #2. 保存到数据库
            session:AsyncSession=request.ctx.session
            async with  session,session.begin():    
                ## 2.1 保存资源ID 
                u1=self.createResourceUUID() 
                opt=DbOperations(session) 
                await self.saveResourceToDb(opt,None,resource_category.hwx,p.record_dir,uid= u1)
                po=p.to_po()  
                po.boxId=device_id
                result= await opt.exist(bizAlarmPO.uuid==po.uuid) 
                if result: 
                    res:m.Response=m.Response.success(message=f"数据“{po.uuid}”重复上传")
                    #通知公众号订阅者
                    Alarm_Upload_View.alarm_success(request,po)
                    log.warn(f"告警信息uuid重复{po.uuid}")
                    return JSON_util.response(res)
                #关联视频资源 视频资源可能为空，有UUID 但没有资源
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
                Alarm_Upload_View.alarm_success(request,po)
                return JSON_util.response(res) 
        except Exception as e:
            log.err(f"上告警失败：{e}")
            res:m.Response=m.Response.fail( message=e)
            return JSON_util.response(res)  