from functools import wraps
from sanic.views import HTTPMethodView  # 基于类的视图
from sanic import Request
from co6co_db_ext.db_utils import db_tools
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations, DbOperations, InstrumentedAttribute
from co6co_sanic_ext.model.res.result import Page_Result
from co6co_sanic_ext.utils import JSON_util
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session
from typing import TypeVar, Dict, List, Any, Tuple
from co6co_sanic_ext.model.res.result import Result, Page_Result
import aiofiles
import os
import multipart
from io import BytesIO
from co6co_web_db.utils import DbJSONEncoder
from sqlalchemy import Select
from co6co_db_ext.po import BasePO, UserTimeStampedModelPO
from datetime import datetime


from co6co.utils import log, getDateFolder
# from api.auth import authorized


class BaseMethodView(HTTPMethodView):
    """
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用
    views.POST  : --> query list
    views.PUT   :---> Add 
    view.PUT    :---> Edit
    view.DELETE :---> del

    """

    def response_json(self, data: Result | Page_Result):
        return DbJSONEncoder.json(data)

    def usable_args(self, request: Request) -> dict:
        """
        去除列表
        request.args={name:['123'],groups:["a","b"]}
        return {name:'123',groups:["a","b"]}
        """
        args: dict = request.args
        data_result = {}
        for key in args:
            value = args.get(key)
            if len(value) == 1:
                data_result.update({key: value[0]})
            else:
                data_result.update({key: value})
        return data_result

    async def save_body(self, request: Request, root: str):
        # 保存上传的内容
        subDir= getDateFolder(format='%Y-%m-%d-%H-%M-%S')
        filePath = os.path.join(root, getDateFolder(), f"{subDir}.data")
        filePath = os.path.abspath(filePath)  # 转换为 os 所在系统路径
        folder = os.path.dirname(filePath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        async with aiofiles.open(filePath, 'wb') as f:
            await f.write(request.body)
        # end 保存上传的内容

    async def parser_multipart_body(self, request: Request) -> Tuple[Dict[str, tuple | Any], Dict[str, multipart.MultipartPart]]:
        """
        解析内容: multipart/form-data; boundary=------------------------XXXXX,
        的内容
        """
        env = {
            "REQUEST_METHOD": "POST",
            "CONTENT_LENGTH": request.headers.get("content-length"),
            "CONTENT_TYPE": request.headers.get("content-type"),
            "wsgi.input": BytesIO(request.body)
        }
        data, file = multipart.parse_form_data(env)
        data_result = {}
        # log.info(data.__dict__)
        for key in data.__dict__.get("dict"):
            value = data.__dict__.get("dict").get(key)
            if len(value) == 1:
                data_result.update({key: value[0]})
            else:
                data_result.update({key: value})
        # log.info(data_result)
        return data_result, file

    async def save_file(self, file, path: str):
        """
        保存上传的文件
        file.name
        """
        async with aiofiles.open(path, 'wb') as f:
            await f.write(file.body)

    async def _save_file(self, request: Request, *savePath: str, fileFieldName: str = None):
        """
        保存上传的文件
        """
        p_len = len(savePath)
        if fileFieldName != None and p_len == 1:
            file = request.files.get(fileFieldName)
            await self.save_file(file, *savePath)
        elif p_len == len(request.files):
            i: int = 0
            for file in request.files:
                file = request.files.get('file')
                await self.save_file(file, savePath[i])
                i += 1

    async def _get_list(self, request: Request, filterItems: absFilterItems, field: InstrumentedAttribute = "*"):
        """
        列表
        """
        filterItems.__dict__.update(request.json)
        async with request.ctx.session as session:
            opt = DbPagedOperations(session, filterItems)
            total = await opt.get_count(field)
            result = await opt.get_paged()
            pageList = Page_Result.success(result)
            pageList.total = total
            await opt.commit()
            return JSON_util.response(pageList)

    async def _del_po(self, request: Request, poType: TypeVar, pk: int):
        """
        删除数据库对象
        """
        async with request.ctx.session as session:
            session: AsyncSession = session
            operation = DbOperations(session)
            po = await operation.get_one_by_pk(poType, pk)
            if po == None:
                return JSON_util.response(Result.fail(message=f"未该'{pk}'对应得数据!"))
            await operation.delete(po)
            await operation.commit()
            return JSON_util.response(Result.success())

    def getFullPath(self, root, fileName: str) -> Tuple[str, str]:
        """
        获取去路径和相对路径
        """
        filePath = "/".join(["", getDateFolder(), fileName])
        fullPath = os.path.join(root, filePath[1:])

        return fullPath, filePath

    def get_db_session(self, request: Request) -> AsyncSession | scoped_session:
        session = request.ctx.session
        if isinstance(session, AsyncSession):
            return session
        elif isinstance(session, scoped_session):
            return session
        raise Exception("未实现DbSession")
        return session

    async def query_mapping(self, request: Request, select: Select, oneRecord: bool = False):
        """
        执行查询: 一个列表|一条记录
        """
        try:
            async with self.get_db_session(request) as session, session.begin():
                executer = await session.execute(select)
                if oneRecord:
                    result = executer.mappings().fetchone()
                    result = Result.success(db_tools.one2Dict(result))
                    return JSON_util.response(result)
                else:
                    result = executer.mappings().all()
                    result = Result.success(db_tools.list2Dict(result))
                    return JSON_util.response(result)
        except Exception as e:
            result = Result.fail(message=f"请求失败：{e}")

    async def query_page(self, request: Request, filter: absFilterItems):
        """
        分页查询
        """
        filter.__dict__.update(request.json)
        try:
            async with self.get_db_session(request) as session, session.begin():
                session: AsyncSession = session
                total = await session.execute(filter.count_select)
                total = total.scalar()
                executer = await session.execute(filter.list_select)
                result = executer.mappings().fetchall()
                result = db_tools.list2Dict(result)
                pageList = Page_Result.success(result, total=total)
                return JSON_util.response(pageList)
        except Exception as e:
            pageList = Page_Result.fail(message=f"请求失败：{e}")
            return JSON_util.response(pageList)

    async def add(self, request: Request, po: BasePO, userId=None, func=None):
        """
        增加
        """
        try:
            po.__dict__.update(request.json)
            async with self.get_db_session(request) as session, session.begin():
                session.add(po)
                if isinstance(po, UserTimeStampedModelPO):
                    po.createTime = datetime.now()
                    po.createUser = userId
                if func != None: func(po)
                session.add(po)
            return JSON_util.response(Result.success())
        except Exception as e:
            return JSON_util.response(Result.fail(message=e))

    async def edit(self, request: Request, pk: any, po: BasePO, poType: TypeVar, userId=None, func=None):
        """
        编辑
        """
        try:
            po.__dict__.update(request.json)
            async with self.get_db_session(request) as session, session.begin():
                oldPo: poType = await session.get_one(poType, pk)
                if oldPo == None:
                    return JSON_util.response(Result.fail(message=f"未查到‘{pk}’对应的信息!"))
                if isinstance(oldPo, UserTimeStampedModelPO):
                    oldPo.updateTime = datetime.now()
                    oldPo.updateUser = userId
                if func != None:
                    func(oldPo, po)
                return JSON_util.response(Result.success())
        except Exception as e:
            return JSON_util.response(Result.fail(message=e))

    async def remove(self, request: Request, pk: any,   poType: TypeVar):
        """
        删除
        """
        try:
            async with self.get_db_session(request) as session, session.begin():
                oldPo: poType = await session.get_one(poType, pk)
                if oldPo == None:
                    return JSON_util.response(Result.fail(message=f"未查到‘{pk}’对应的信息!"))
                await session.delete(oldPo)
                return JSON_util.response(Result.success())
        except Exception as e:
            return JSON_util.response(Result.fail(message=e))


"""
class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 
"""
