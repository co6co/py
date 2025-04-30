
from sanic.response import text, file, raw, json
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Update, Insert
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import DevicePO
from view_model._filters.device import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute

import pandas as pd
from co6co.utils import log
from io import BytesIO
from datetime import datetime
import ipaddress
from model.enum import DeviceCategory
from co6co.utils import try2int
from typing import TypedDict
from view_model import ImportView


class columnsMap(TypedDict):
    code: str
    category: str
    name: str
    ip: str
    lng: str
    lat: str
    state: str
    username: str
    password: str


class ExistView(AuthMethodView):
    routePath = ObjectExistRoute

    async def get(self, request: Request, code: str, pk: int = 0):
        result = await self.exist(request, DevicePO.code == code, DevicePO.id != pk)
        return exist(result, "编码编码", code)


class DeviceCategoryView(AuthMethodView):
    routePath = "/category"

    async def get(self, request: Request):
        """
        树形选择下拉框数据
        """
        return JSON_util.response(Result.success(DeviceCategory.to_dict_list()))


class Views(AuthMethodView):
    async def get(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        select = (
            Select(DevicePO.id, DevicePO.name, DevicePO.code, DevicePO.state)
            .order_by(DevicePO.code.asc())
        )
        return await self.query_list(request, select,  isPO=False)

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()

        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = DevicePO()
        userId = self.getUserId(request)
        po.__dict__.update(request.json)

        async def before(po: DevicePO, session: AsyncSession, request):
            exist = await db_tools.exist(session,  DevicePO.code.__eq__(po.code), column=DevicePO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))

        return await self.add(request, po, json2Po=False, userId=userId, beforeFun=before)


class columnsMap:
    def __init__(self, codeIndex: int, categoryIndex: int, nameIndex: int, ipIndex: int, lngIndex: int, latIndex: int, stateIndex: int, userNameIndex: int, passwordIndex: int):
        self.codeIndex = codeIndex
        self.categoryIndex = categoryIndex
        self.nameIndex = nameIndex
        self.ipIndex = ipIndex
        self.lngIndex = lngIndex
        self.latIndex = latIndex
        self.stateIndex = stateIndex
        self.userNameIndex = userNameIndex
        self.passwordIndex = passwordIndex


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        async def before(oldPo: DevicePO, po: DevicePO, session: AsyncSession, request):
            exist = await db_tools.exist(session, DevicePO.id != oldPo.id, DevicePO.code.__eq__(po.code), column=DevicePO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"'{po.code}'已存在！"))
        return await self.edit(request, pk, DevicePO,  userId=self.getUserId(request), fun=before)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, DevicePO)


class DeviceImportView(ImportView):

    def __init__(self):
        sheet_name = "设备列表"
        super().__init__(sheet_name, "template.xlsx")

    def columns(self):
        code = "编号"
        category = f"设备类型({DeviceCategory.to_labels_str()})"
        name = "名称"
        ip = "IP地址"
        lng = "经度"
        lat = "纬度"
        state = "状态(1:启用 0:禁用)"
        username = "用户名（可空）"
        password = "密码（可空）"
        columns = [code, category, name, ip, lng, lat, state, username, password]
        col_type_mapping = {
            code: "codeIndex",
            category: "categoryIndex",
            name: "nameIndex",
            ip: "ipIndex",
            lng: "lngIndex",
            lat: "latIndex",
            state: "stateIndex",
            username: "userNameIndex",
            password: "passwordIndex"
        }
        col_dict = {col_type_mapping[col]: idx for idx, col in enumerate(columns) if col in columns}
        # 动态生成 TypedDict
        # ColMap = TypedDict('ColMap', {v: int for v in col_dict.values()})
        # colmap = ColMap(**col_dict)
        self.columnsMap = columnsMap(**col_dict)
        return columns

    def paraserIP(self, ip):
        try:
            ip = ipaddress.IPv4Address(ip)
            return str(ip)
        except ValueError:
            return None

    async def handler(self,  item: list[list], dbSession: AsyncSession, request: Request, **kvags):
        """
        导入数据
        """
        userId = self.getUserId(request)
        colMap = self.columnsMap
        if item:
            try:
                # nan!=nan
                item = ['' if x != x else x for x in item]
                code = str(item[colMap.codeIndex])
                category = try2int(item[colMap.categoryIndex])
                name = str(item[colMap.nameIndex])
                ip = self.paraserIP(item[colMap.ipIndex])
                lng = str(item[colMap.lngIndex])
                lat = str(item[colMap.latIndex])
                state = try2int(item[colMap.stateIndex], 1)
                username = str(item[colMap.userNameIndex])
                password = str(item[colMap.passwordIndex])
                if not ip:
                    msg = f"IP地址{ip}格式错误"
                    return -1
                exist = await db_tools.exist(dbSession, DevicePO.ip.__eq__(ip), column=DevicePO.id)
                if exist:
                    # print(f"设备{ip}已存在", exist)
                    updateSml = Update(DevicePO).filter(DevicePO.ip.__eq__(ip)).values(
                        {
                            DevicePO.code: code,
                            DevicePO.category: category,
                            DevicePO.name: name,
                            DevicePO.lng: lng,
                            DevicePO.lat: lat,
                            DevicePO.state: state,
                            DevicePO.userName: username,
                            DevicePO.passwd: password,
                            DevicePO.updateTime: datetime.now(),
                            DevicePO.updateUser: userId
                        }
                    )
                    result = await db_tools.execSQL(dbSession, updateSml)
                    if result > 0:
                        return 2
                else:
                    InsertSml = Insert(DevicePO).values(
                        {
                            DevicePO.code: code,
                            DevicePO.name: name,
                            DevicePO.category: category,
                            DevicePO.ip: ip,
                            DevicePO.lng: lng,
                            DevicePO.lat: lat,
                            DevicePO.state: state,
                            DevicePO.userName: username,
                            DevicePO.passwd: password,
                            DevicePO.createTime: datetime.now(),
                            DevicePO.createUser: userId
                        }
                    )
                    result = await db_tools.execSQL(dbSession, InsertSml)
                    if result > 0:
                        return 1
            except Exception as e:
                log.error(f"导入数据失败，{e}")
                return -1
        return 0

    async def post(self, request: Request):
        ccc = self.columns()  # for self.columnsMap
        return await self._post(request, "split")
