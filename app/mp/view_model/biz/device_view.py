

from sanic import Request,redirect

from sqlalchemy.ext.asyncio import AsyncSession

from co6co_sanic_ext.utils import JSON_util, sys_json
from co6co.utils import log
from model.filters.DeviceFilterItems import DeviceFilterItems, CameraFilterItems
from sqlalchemy.orm import joinedload
from view_model.base_view import AuthMethodView
from model.pos.biz import  bizCameraPO
from model.enum import device_type
from model.params.devices import cameraParam, streamUrl
from co6co_sanic_ext.model.res.result import Result, Page_Result
from co6co_db_ext .db_utils import db_tools
from sqlalchemy.sql import Select

from co6co.utils import hash
import os
from utils import createUuid
import datetime
from sanic.response import file, empty, json
from sqlalchemy.engine.row import RowMapping

from view_model.biz.poster_view import Image_View



class Devices_View(AuthMethodView):
    """
    设备API
    """
    async def get(self, request: Request):
        """
        获取设备类型
        """
        dictList = device_type.to_dict_list()
        return JSON_util.response(Result.success({"categoryList": dictList,"cameraCategory":device_type.ip_camera.val,"boxCategory":device_type.box.val}))

    async def post(self, request: Request):
        """
        获取设备 list 
        """
        param = DeviceFilterItems()
        param.__dict__.update(request.json)
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer = await session.execute(param.count_select)
            total = executer.scalar()
            result = await session.execute(param.list_select)
            result = result.mappings().all()
            result = [dict(a) for a in result]
            pageList = Page_Result.success(result, total=total)
            await session.commit()
        return JSON_util.response(pageList)

class Device_Config_View(AuthMethodView):
    async def camera(self,request: Request,deviceId:int):
        async with request.ctx.session as session:
            session: AsyncSession = session
            select=(Select(bizCameraPO).filter(bizCameraPO.id==deviceId))
            executer = await session.execute(select)
            po :bizCameraPO= executer.fetchone() 
            if po ==None: return redirect("/",404)
            else:
                #https://www.jshwx.com.cn/webshell/page/webssh.html?dOption=eyJwYXNzd29yZCI6Imh3eDEyMyIsIm9wZXJhdGUiOiJjb25uZWN0IiwicG9ydCI6IjIyIiwiaG9zdCI6IjEwLjYuMS4xODEiLCJkZXZuYW1lIjoi6ZW/5rGfNjIwMzMiLCJkZXZtb2RlbCI6IkhZQ1lfMDYiLCJ1c2VybmFtZSI6InJvb3QiLCJuZ3Jva1Rva2VuIjoiMDVmZjNhNmVjYzIzM2UwZSJ9
                dict={"operate":"connect","host":po.ip,"username":"root","password":"hwx123","decname":po.name,"devmodel":'',"ngrokToken":""}
                v=hash.enbase64(f"{dict}")
                return redirect (f"https://www.jshwx.com.cn/webshell/page/webssh.html?dOption={v}")



    async def get(self, request: Request,type:device_type,deviceId:int):
        """
        跳转到设备配置
        """
        if(type==device_type.ip_camera) : return await self.camera(request,type,deviceId)
        if(type==device_type.box):return redirect("/",404)
        if(type==device_type.router):return redirect("/",404)


class IP_Cameras_View(AuthMethodView): 
    async def put(self, request: Request):
        """
        增加相机
        如果没有可以删除
        """
        try:
            param = cameraParam()
            param.__dict__.update(request.json)
            id=self.getUserId(request)
            async with request.ctx.session as session:
                session: AsyncSession = session
                po = bizCameraPO()
                param.set_po(po) 
                po.uuid=createUuid() 
                po.createTime=datetime.datetime.now() 
                po.createUser=id 
                session.add(po)
                await session.commit()
            return JSON_util.response(Result.success())
        except Exception as e:
            return JSON_util.response(Result.fail(message=e))

    async def post(self, request: Request):
        """
        获取相机设备 list 
        """
        param = CameraFilterItems()
        param.__dict__.update(request.json)
        
        async with request.ctx.session as session:
            session: AsyncSession = session
            executer = await session.execute(param.count_select)
            total = executer.scalar()
            result = await session.execute(param.list_select)
            #result = result.mappings().all() 
            #result = [dict(a) for a in result]
            result = result.scalars().all()
            result=db_tools.remove_db_instance_state(result)
            pageList = Page_Result.success(result, total=total)
            await session.commit()
        return JSON_util.response(pageList)


class IP_Camera_poster_View(Image_View):
    async def get(self, request: Request, pk: int):
        """
        相机 poster 
        """
        async with request.ctx.session as session:
            session: AsyncSession = session
            one: bizCameraPO = await session.get_one(bizCameraPO, pk)
            if one != None and one.poster != None and os.path.exists(one.poster):
                return await file(one.poster, mime_type="image/jpeg")
            await session.commit()
        return empty(status=404)


class IP_Camera_View(AuthMethodView): 
    async def put(self, request: Request, pk: int):
        """
        编辑相机
        如果没有可以删除
        """
        try:
            param  = cameraParam()
            param.__dict__.update(request.json)
            id=self.getUserId(request)
            async with request.ctx.session as session,session.begin():
                session: AsyncSession = session
                select = (
                    Select(bizCameraPO).filter(bizCameraPO.id==pk)
                )
                execute=await session.execute(select) 
                po:bizCameraPO=execute.scalar()
                if po == None: return JSON_util.response(Result.fail(message="未找到设备!")) 
                param.set_po(po)
                po.updateTime=datetime.datetime.now()  
            return JSON_util.response(Result.success())
        except Exception as e: 
            log.err(f"编辑相机失败：{e}")
            return JSON_util.response(Result.fail(message=e))
    async def delete(self, request: Request, pk: int):
        """
        删除相机
        """
        log.succ(f"删除相机：{pk}")
        try: 
            async with request.ctx.session as session,session.begin():
                session: AsyncSession = session
                select = ( Select(bizCameraPO).filter(bizCameraPO.id==pk) )
                execute=await session.execute(select) 
                po:bizCameraPO=execute.scalar()
                if po == None: return JSON_util.response(Result.fail(message="未找到设备!"))
                await session.delete(po)  
                return JSON_util.response(Result.success())
        except Exception as e: 
            log.err(f"删除相机{pk}失败：{e}")
            return JSON_util.response(Result.fail(message=e))


    