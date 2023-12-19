
from co6co_db_ext .db_operations import DbOperations, DbPagedOperations, and_, joinedload
from sanic import Request
from sanic.response import text, raw

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from co6co_sanic_ext.utils import JSON_util
import json
from co6co.utils import log
from model.filters.DeviceFilterItems import CameraFilterItems

from view_model.base_view import AuthMethodView
from model.pos.biz import bizDevicePo,bizCameraPO
from co6co_sanic_ext.model.res.result import Page_Result
from sqlalchemy.sql import Select

from model.enum import device_type
import os,cv2,datetime
from sanic.response import file,empty

from view_model.biz.poster_view import Image_View
class IP_Cameras_View(AuthMethodView):

    async def post(self, request: Request):
        """
        获取相机设备 list
        
         未知原因 dbsession 会卡住
        """
        param = CameraFilterItems()
        param.__dict__.update(request.json) 
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer=  await session.execute(param.count_select)
            total=executer.scalar() 
            result = await session.execute(param.list_select)
            result = result.mappings().all()
            result = [dict(a) for a in result] 
            pageList = Page_Result.success(result, total=total) 
            await session.commit()
        return JSON_util.response(pageList)


class IP_Camera_View(Image_View):
    async def get(self, request: Request,pk:int):
        """
        相机 poster 
        """   
        async with request.ctx.session as session:
            session: AsyncSession = session
            one: bizCameraPO= await session.get_one(bizCameraPO,pk)
            if one!=None and one.poster !=None and os.path.exists(one.poster) :
                return await file(one.poster,mime_type="image/jpeg")  
            await session.commit() 
        return empty(status=404)
