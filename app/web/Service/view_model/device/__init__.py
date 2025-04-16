
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


class ExistView(AuthMethodView):
    routePath = ObjectExistRoute

    async def get(self, request: Request, code: str, pk: int = 0):
        result = await self.exist(request, DevicePO.code == code, DevicePO.id != pk)
        return exist(result, "编码编码", code)


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
        param.__dict__.update(request.json)

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


class ImportView(AuthMethodView):
    routePath = "/import"

    def __init__(self):
        self.sheet_name = "设备列表"

    def template(self):
        # 假设模板文件名为 template.xlsx，放在当前目录下
        columns = ['编号', '名称', "IP地址", "经度", "维度", "状态 1:启用 0:禁用", "用户名（可空）", "密码（可空）"]
        df = pd.DataFrame(columns=columns)
        # new_data = [
        #    [1, '小明', 95],
        #    [2, '小红', 88]
        # ]
        # df = pd.DataFrame(new_data, columns=columns)
        # df = pd.concat([df, new_df], ignore_index=True)
        # 创建一个内存中的二进制流
        output = BytesIO()
        # 将 DataFrame 写入二进制流，这里以 Excel 格式为例
        df.to_excel(output, index=False, sheet_name=self.sheet_name)
        # 移动到流的起始位置
        output.seek(0)
        # 获取二进制数据
        binary_data = output.read()
        template_path = 'template.xlsx'
        # df.to_excel(template_path, index=False)
        return binary_data, template_path

    async def head(self, request: Request):
        """
        文件
        """
        binary_data, fileName = self.template()
        return await self.response_head(len(binary_data), fileName)

    async def get(self, request: Request):
        """
        获取模板文件
        """
        try:
            binary_data, template_path = self.template()
            headers = {
                "Content-Disposition": f'attachment; filename="{template_path}"'
            }
            # return await file(template_path, filename='template.xlsx')
            return raw(binary_data, headers=headers)
        except FileNotFoundError:
            return self.response_json(Result.fail(message="Template file not found"))
        except Exception as e:
            return self.response_json(Result.fail(message=f"Err:{e}"))

    def paraserIP(self, ip):
        try:
            ip = ipaddress.IPv4Address(ip)
            return str(ip)
        except ValueError:
            return None

    async def importData(self, request: Request, data: list[list]):
        """
        导入数据
        """
        dbSession = self.get_db_session(request)
        uploadNum = 0
        insertNum = 0
        errorNum = 0
        msg = ""
        userId = self.getUserId(request)
        if data:
            try:
                for item in data:
                    # nan!=nan
                    item = ['' if x != x else x for x in item]
                    code = str(item[0])
                    name = str(item[1])
                    ip = self.paraserIP(item[2])
                    lng = str(item[3])
                    lat = str(item[4])
                    state = int(item[5])

                    username = str(item[6])
                    password = str(item[7])
                    if not ip:
                        msg = f"IP地址{ip}格式错误"
                        errorNum += 1
                        continue
                    exist = await db_tools.exist(dbSession, DevicePO.ip.__eq__(ip), column=DevicePO.id)
                    if exist:
                        # print(f"设备{ip}已存在", exist)
                        updateSml = Update(DevicePO).filter(DevicePO.ip.__eq__(ip)).values(
                            {
                                DevicePO.code: code,
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
                            uploadNum += 1
                    else:
                        InsertSml = Insert(DevicePO).values(
                            {
                                DevicePO.code: code,
                                DevicePO.name: name,
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
                            insertNum += 1
                await dbSession.commit()
            except Exception as e:
                msg = str(e)
                log.err(f"导入数据出现错误，{e}")
                await dbSession.rollback()
        return insertNum, uploadNum, errorNum, msg

    async def post(self, request: Request):
        try:
            # 检查请求中是否包含文件
            if 'file' not in request.files:
                return self.response_json(Result.fail(message="No file part"))
            file = request.files.get('file')

            filename: str = file.name
            # 检查文件是否为 Excel 格式
            if not filename.endswith(('.xlsx', '.xls')):
                return self.response_json(Result.fail(message="Invalid file format"))

            # 读取 Excel 文件
            df = pd.read_excel(file.body, sheet_name=self.sheet_name)
            # 这里可以对数据进行进一步处理
            data = df.to_dict(orient='split')  # index,columns,data
            itemList = data.get('data')
            log.warn(f"导入数据：{itemList}")
            insertNum, uploadNum, errorNUm, msg = await self.importData(request, itemList)
            if insertNum+uploadNum > 0:
                return self.response_json(Result.success(message=f"导入成功，新增{insertNum}条，更新{uploadNum}条，错误{errorNUm}条,{msg}"))
            else:
                return self.response_json(Result.fail(message=f"导入数据为0,可能执行失败失败，{msg}"))
        except Exception as e:
            return self.response_json(Result.fail(message=f"导入出现错误，{e}"))
