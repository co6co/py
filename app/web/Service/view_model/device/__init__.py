
from sanic.response import text, file, raw, json
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import DevicePO
from view_model._filters.device import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute
from services.tasks import custom
import pandas as pd
from co6co.utils import log
from io import BytesIO


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

    def template(self):
        # 假设模板文件名为 template.xlsx，放在当前目录下
        columns = ['编号', '名称', "网络地址", "经度", "维度", "状态 1:启用 0:禁用", "用户名（可空）", "密码（可空）"]
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
        df.to_excel(output, index=False)
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

    async def post(self, request: Request):
        try:
            # 检查请求中是否包含文件
            if 'file' not in request.files:
                return self.response_json(Result.fail(message="No file part"))
            file = request.files.get('file')
            # 检查文件是否为 Excel 格式
            if not file.name.endswith(('.xlsx', '.xls')):
                return self.response_json(Result.fail(message="Invalid file format"))

            # 读取 Excel 文件
            df = pd.read_excel(file.body)
            # 这里可以对数据进行进一步处理
            data = df.to_dict(orient='records')
            log.info(data)
            return self.response_json(Result.fail(message="Invalid file format"))
        except Exception as e:
            return self.response_json(Result.fail(message=f"导入出现错误，{e}"))
