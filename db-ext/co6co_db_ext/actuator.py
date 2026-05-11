from .db_filter import absFilterItems
from sqlalchemy.ext.asyncio import AsyncSession
from .db_utils import db_tools,  DbCallable, QueryOneCallable, UpdateOneCallable, QueryListCallable, QueryPagedByFilterCallable
from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from co6co.data.result import Result,Page_Result
from typing import Callable, Any, Dict , List ,Tuple
from co6co.utils import log
from functools import wraps
from sanic.views import HTTPMethodView  # 基于类的视图
from sanic import Request
from co6co_db_ext.db_utils import db_tools, QueryPagedByFilterCallable
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations, DbOperations, InstrumentedAttribute
from co6co_sanic_ext.model.res.result import Page_Result
from co6co_sanic_ext.utils import JSON_util
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session
from typing import TypeVar, Dict, List, Any, Tuple, Optional, Callable
from co6co_sanic_ext.model.res.result import Result, Page_Result

from io import BytesIO
from co6co_web_db.utils import DbJSONEncoder
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute

from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from co6co_sanic_ext .view_model import BaseView
from co6co_db_ext.po import BasePO, TimeStampedModelPO, UserTimeStampedModelPO, CreateUserStampedModelPO
from datetime import datetime
from co6co.utils.tool_util import list_to_tree, get_current_function_name
from co6co_db_ext.db_utils import db_tools,  DbCallable, QueryOneCallable, UpdateOneCallable, QueryListCallable, QueryPagedByFilterCallable

from co6co_web_session.base import SessionDict
from multiprocessing.managers import DictProxy


from co6co.utils import log, getDateFolder

class actuator:
    def __init__(self, session: AsyncSession, filterItem: absFilterItems):
        self.session=session
        self.filter=filterItem
        pass
    async def get_one(self, select: Select, isPO: bool = True):
        """
        获取一条记录
        """
        call = QueryOneCallable(self.session)
        return await call(select, isPO)
    async def get_one(self,   select: Select, isPO: bool = True, remove_db_instance: bool = True, resultHanlder: Callable[[Any], Any] = None):
        """
        从数据库中获取一个对象
        resultHanlder: 不为空是，返回值将作为最终的返回结果
              使有机会改变从数据库中查询的结果              
        """
        result = await self.get_one(select, isPO)
        if isPO and remove_db_instance:
            result = db_tools.remove_db_instance_state(result)
        if resultHanlder  :
            bckResult = resultHanlder(result)
            if bckResult  :
                result = bckResult
        if  result is None:
            return Result.fail(message="未查询到数据")
        else:
            return self.response_json(Result.success(result))

    async def update_one(self,  select: Select, editFn: None):
        """
        更新 PO
        editFn(session,basePO): 对select 的第一条记录赋值,没有赋值或者没有更改->不执行update
                                返回一个对象:http 应答，
                                        None:滚数据请求失败

        """
        try:
            call = UpdateOneCallable(self.session)
            result = await call(select, editFn)
            if result != None:
                return result
            else:
                return Result.fail(message="更新失败")
        except Exception as e:
            return   Result.fail(message=f"更新失败{e}")

    async def query_mapping(self,  select: Select, oneRecord: bool = False):
        """
        执行查询: 一个列表|一条记录
        """
        try:
            async with self.session as session, session.begin():
                executer = await session.execute(select)
                if oneRecord:
                    result = executer.mappings().fetchone()
                    result = Result.success(db_tools.one2Dict(result))
                    return  result
                else:
                    result = executer.mappings().all()
                    result = Result.success(db_tools.list2Dict(result))
                    return  result
        except Exception as e:
            return   Result.fail(message=f"查询失败{e}")

    async def _query(self,  select: Select, isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询: 列表 
        """
        try:
            session: AsyncSession = self.session
            query = QueryListCallable(session)
            data = await query(select, isPO, remove_db_instance, param)
            result = db_tools.list2Dict(data)
            return result
        except Exception as e:
            log.error(f"查询失败{e}")
        except Exception as e:
            return None

    async def exist(self,    *filters: ColumnElement[bool], column: InstrumentedAttribute = "*"):
        """
        查看对象是否操作
        """
        return await db_tools.exist(self.session, *filters, column=column)

    async def query_list(self,  select: Select,   isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询:  列表 
        """
        try:
            result = await self._query( select,   isPO, remove_db_instance, param)
            return result
        except Exception as e:
            log.err(f"查询失败{e}",e)
            return None

    async def query_tree(self,  select: Select, rootValue: any = None, pid_field: str = "pid", id_field: str = "id", isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询: tree列表 
        """
        try:
            result = await self._query( select,   isPO, remove_db_instance, param)
            if result == None:
                treeList = []
            else:
                treeList = list_to_tree(result, rootValue, pid_field, id_field)
            if len(treeList) == 0:
                return  []
            return treeList
        except Exception as e:
            log.error(f"查询失败{e}",e) 

    async def query_page(self,  isPO: bool = True, remove_db_instance=True):
        """
        分页查询
        """
         
        try:
            query = QueryPagedByFilterCallable(self.session)
            total, result = await query(self.filterItem, isPO, remove_db_instance)
            pageList = Page_Result.success(result, total=total)
            return JSON_util.response(pageList)
        except Exception as e:
            log.error(f"查询失败{e}",e)
            return None

    async def execSqls(self,  *sml: Update | Delete | Insert, callBck=None, smlParamList: List[Dict | Tuple | List] = None):
        callable = DbCallable(self.session)

        async def exec(session: AsyncSession):
            try:
                result = []
                index = 0
                for sql in sml:

                    param = None
                    if smlParamList != None and len(smlParamList) == len(sml):
                        param = smlParamList[index]
                    r = await db_tools.execSQL(session, sql, param)
                    result.append(r)
                    index += 1
                if callBck != None:
                    return await callBck(*result)
            except Exception as e:
                await session.rollback()
                log.error(f"执行SQL失败{sql} {e}",e)
                return None

        return await callable(exec)

    async def batchAdd(self,  poList: List[BasePO],   userId=None, beforeFun: Callable[[BasePO, AsyncSession, Request], None | Any] = None, afterFun: Callable[[List[BasePO], AsyncSession, Request], None] = None):
        try:

            async def exec(session: AsyncSession):
                for po in poList:
                    po.add_assignment(userId)
                    if beforeFun != None:
                        result = await beforeFun(po, session)
                        if result != None:
                            await session.rollback()
                            return result
                session.add_all(poList)
                if afterFun != None:
                    session.flush()
                    await afterFun(poList, session)
                return True

            callable = DbCallable(self.session)
            return await callable(exec)
        except Exception as e:
            log.error(f"批量增加失败{e}",e)
            return None

    async def add(self,  po: BasePO, json2Po: bool = True, userId=None, beforeFun: Callable[[BasePO, AsyncSession], None | Any] = None, afterFun: Callable[[BasePO, AsyncSession], None] = None):
        """
        增加 

        request: Request, 
        po: BasePO,      #实体类对象 
        userId=None, # 用户ID
        beforeFun(po, session, request),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(po, session, request),     # 可在实体中获取 自增id

        return JSONResponse
        """
        try:
            if json2Po:
                po.__dict__.update(request.json)

            async def exec(session: AsyncSession):
                po.add_assignment(userId)

                if beforeFun != None:
                    result = await beforeFun(po, session, request)
                    if result != None:
                        await session.rollback()
                        return result
                session.add(po)
                if afterFun != None:
                    session.flush()
                    await afterFun(po, session, request)
                return JSON_util.response(Result.success())

            callable = DbCallable(self.session)
            return await callable(exec)
        except Exception as e:
            return self.response_error(request, e)

    async def edit(self, request: Request, pkOrSelect:  int | str | Select, poType: TypeVar, po: Optional[BasePO] = None, userId=None, fun=None, json2Po: bool = True):
        """
        编辑

        request: Request, 
        pk: any,          # 主键
        poType: TypeVar,  # 实体类型
        po:BasePO    ,    # None:根据传入的 poType创建,用 request.json赋值
        userId=None, # 用户ID
        fun=None,    # 执行一些其他操作，返回值将直接返回客户端并且回滚数据库操作
        json2Po:bool # 根据 请求的json 转换的对象更新 实体对象，在 fun 之前执行

        return JSONResponse
        """
        try:
            if po == None:
                po = poType()
                po.__dict__.update(request.json)
            call = DbCallable(self.session)

            async def exec(session: AsyncSession):
                oldPo: BasePO = None
                if isinstance(pkOrSelect, Select):
                    oldPo = await db_tools.execForPo(session, pkOrSelect, remove_db_instance_state=False)
                else:
                    oldPo: BasePO = await session.get_one(poType, pkOrSelect)
                if oldPo == None:
                    return JSON_util.response(Result.fail(message=f"未查到‘{pk}’对应的信息!"))
                oldPo.edit_assignment(userId)
                if json2Po:
                    oldPo.update(po)
                if fun != None:
                    result = await fun(oldPo, po, session, request)
                    if result != None:
                        await session.rollback()
                        return result
                return JSON_util.response(Result.success())
            return await call(exec)
        except Exception as e:
            return self.response_error(request, e)

    async def remove(self, request: Request, pk: any, poType: TypeVar,  beforeFun=None, afterFun=None):
        """
        删除 

        request: Request, 
        pk: any,      #主键值
        poType: TypeVar # 实体类型
        beforeFun(oldPo, session),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(oldPo, session, request),     # 返回值将直接返回客户端，回滚数据库操作

        return JSONResponse
        """
        try:
            async with self.session as session, session.begin():
                oldPo: BasePO = await session.get_one(poType, pk)
                if oldPo == None:
                    return JSON_util.response(Result.fail(message=f"未找到‘{pk}’对应的信息!"))
                if beforeFun != None:
                    result = await beforeFun(oldPo, session)
                    if result != None:
                        await session.rollback()
                        return result
                await session.delete(oldPo)
                if afterFun != None:
                    result = await afterFun(oldPo, session, request)
                    if isinstance(result, Result):
                        return JSON_util.response(result)
                    elif result != None:
                        await session.rollback()
                        return result
                return JSON_util.response(Result.success())
        except Exception as e:
            await session.rollback()
            return self.response_error(request, e)

	