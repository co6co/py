from sqlalchemy.ext.asyncio import AsyncSession
from .po import BasePO, TimeStampedModelPO, UserTimeStampedModelPO, CreateUserStampedModelPO
from .db_filter import absFilterItems
from .db_utils import db_tools,  DbCallable, QueryOneCallable, UpdateOneCallable, QueryListCallable, QueryPagedByFilterCallable
from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from typing import Callable, Any, Dict , List ,Tuple
from functools import wraps
 
from co6co.data.result import Result,Page_Result
from co6co.utils import log, getDateFolder

from co6co_db_ext.db_utils import db_tools, QueryPagedByFilterCallable
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations, DbOperations, InstrumentedAttribute

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session
from typing import TypeVar, Dict, List, Any, Tuple, Optional, Callable


from io import BytesIO
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute

from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from datetime import datetime
from co6co.utils.tool_util import list_to_tree, get_current_function_name



from multiprocessing.managers import DictProxy

 
# 2. 定义泛型：限定必须是 BasePO 的子类
POType = TypeVar("POType", bound=BasePO)


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
            return  Result.fail(message=f"查询失败{e}")

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
            log.err(f"查询失败{e}")
        except Exception as e:
            return None

    async def exist(self,  *filters: ColumnElement[bool], column: InstrumentedAttribute = "*"):
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
            log.err(f"查询失败{e}",e) 

    async def query_page(self,  isPO: bool = True, remove_db_instance=True):
        """
        分页查询
        """
         
        try:
            query = QueryPagedByFilterCallable(self.session)
            total, result = await query(self.filter, isPO, remove_db_instance)
            pageList = Page_Result.success(result, total=total)
            return pageList
        except Exception as e:
            log.err(f"查询失败{e}",e)
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
                log.err(f"执行SQL失败{sql} {e}",e)
                return None

        return await callable(exec)

    async def batchAdd(self,  poList: List[BasePO],   userId=None, checkFun:Callable[[POType,AsyncSession], Any]  =None, afterFun: Callable[[List[BasePO], AsyncSession,Any], Result] = None):
        try:

            async def exec(session: AsyncSession):
                for po in poList:
                    po.add_assignment(userId)
                    result=None
                    if checkFun != None:
                        result = await checkFun(po, session)
                       
                session.add_all(poList)
                if afterFun != None:
                    session.flush()
                    await afterFun(poList, session,result)
                return True

            callable = DbCallable(self.session)
            return await callable(exec)
        except Exception as e:
            log.err(f"批量增加失败{e}",e)
            return None

    async def add(self,  po: BasePO, json2Po: bool = True, userId=None,  checkFun:Callable[[POType,AsyncSession], Any]  =None, afterFun: Callable[[BasePO, AsyncSession,Any], Result] = None):
        """
        增加 

        request: Request, 
        po: BasePO,      #实体类对象 
        userId=None, # 用户ID
        beforeFun(po, session ),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(po, session ),     # 可在实体中获取 自增id

        return Result
        """
        try:
             
            async def exec(session: AsyncSession):
                po.add_assignment(userId)
                result=None
                if checkFun != None:
                    result = await checkFun(po, session ) 
                session.add(po)
                result=Result.success()
                if afterFun != None:
                    session.flush()
                    result= await afterFun(po, session,result )
                return result

            callable = DbCallable(self.session)
            return await callable(exec)
        except Exception as e:
            log.err(f"执行add err {e}",e)
            return Result.fail(message=f"增加失败{e}")
            

    async def edit(self,  select:   Select,  po: POType , userId=None, fun:Callable[[POType,POType, AsyncSession],Result]=None, json2Po: bool = True):
        """
        编辑

        request: Request, 
        pk: any,          # 主键
        poType: TypeVar,  # 实体类型
        po:BasePO    ,    # None:根据传入的 poType创建,用 request.json赋值
        userId=None, # 用户ID
        fun=None,    # 执行一些其他操作， (oldPO,po,session)
                     # 返回值将直接返回客户端并且回滚数据库操作
        json2Po:bool # 根据 请求的json 转换的对象更新 实体对象，在 fun 之前执行

        return JSONResponse
        """
        try: 
            call = DbCallable(self.session) 
            async def exec(session: AsyncSession):
                oldPo:BasePO = await db_tools.execForPo(session, select, remove_db_instance_state=False)
                #oldPo: BasePO = await session.get_one(POType, pkOrSelect)
                if oldPo == None:
                    return   Result.fail(message=f"未对应的信息!")
                oldPo.edit_assignment(userId)
                if json2Po:
                    oldPo.update(po)
                if fun :
                    return await fun(oldPo, po, session )
                     
                return  Result.success()
            return await call(exec)
        except Exception as e:
            log.err(f"执行edit失败{e}",e)

    async def remove(self,  pk: any, poType: POType,  checkFun:Callable[[POType,AsyncSession], Any]  =None  , afterFun:Callable[[POType,AsyncSession,Any], Result]=None):
        """
        删除 

        request: Request, 
        pk: any,      #主键值
        poType: TypeVar # 实体类型
        checkFun(oldPo, session),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
                    返回值 Result 返回
                    返回   其他   回滚
                    返回   None   继续
        @afterFun afterFun(oldPo, session, await checkFun())   
        return Result
        """
        
        try:
            async with self.session as session, session.begin():
                oldPo: BasePO = await session.get_one(poType, pk)
                if oldPo == None:
                    return Result.fail(message=f"未找到‘{pk}’对应的信息!")
                result=None
                if checkFun != None:
                    result = await checkFun(oldPo, session) 
                await session.delete(oldPo) 
                if afterFun != None:
                    return await afterFun(oldPo, session ,result) 
                return Result.success()
        except Exception as e:
            await session.rollback()
            log.err(f"执行remove失败{e}",e)
            

	
