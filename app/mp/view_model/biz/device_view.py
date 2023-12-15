
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sanic import  Request 
from sanic.response import text,raw

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from co6co_sanic_ext.utils import JSON_util
import json
from co6co.utils import log

from view_model.base_view import  AuthMethodView
from model.pos.biz import bizDevicePo 
from co6co_sanic_ext.model.res.result import Page_Result 
from sqlalchemy.sql import Select

from model.enum import device_type


class IP_Camera_View(AuthMethodView):
   
    async def post(self,request:Request):
        """
        获取相机设备 list
        """
        async with request.ctx.session as session:  
            session:AsyncSession=session 
            opt=DbOperations(session)  
            select=(
                 Select(bizDevicePo).where(bizDevicePo.deviceType==device_type.ip_camera.val) 
            )  
            log.err(type(select))
            result= await opt._get_list(select,True) 
            select=(
                Select( func.count( )).select_from(
                    Select(bizDevicePo).where(bizDevicePo.deviceType==device_type.ip_camera.val) 
                )
            ) 
            total= await opt._get_scalar(select)  
            pageList=Page_Result.success(result,total=total)  
            await session.commit() 
        return JSON_util.response(pageList)