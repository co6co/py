from __future__ import annotations
 
from .po import BasePO 
from .db_filter import absFilterItems

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.row import Row, RowMapping
from sqlalchemy.orm import Mapper

from typing import Callable, Any, Dict , List ,Tuple,Awaitable
from functools import wraps
 
from co6co.data.result import Result,Page_Result
from co6co.utils import log 
from co6co.utils.tool_util import list_to_tree,get_current_function_name
  
from sqlalchemy import func, text,and_
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute 

from typing import TypeVar, Iterator,Dict, List, Any, Tuple, Optional, Callable 

 


# 定义泛型：限定必须是 BasePO 的子类
POType = TypeVar("POType", bound=BasePO) 
class OperationType:
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE" 


def errorLog( module: str, method: str,errorCode: str,e:Exception):
    log.err(f"执行出错所属模块:{module}.{method} Error,错误码:{errorCode}",e)

class Actuator:
    def _error0(self,e: Exception):
        """
        响应错误 message
        """ 
        #debug()
        errorCode =hex(e.__hash__()) 
        errorLog( self.__class__, get_current_function_name(),errorCode,e)
        return Result.fail(message=f"执行不成功，{errorCode}")
    def __init__(self, session: AsyncSession):
        """
            # 查询1：选择特定列 返回元组
            stmt1 = select(userPO.id, userPO.name)
            result1 = await session.execute(stmt1)
            # 方式1：获取元组列表
            rows1 = result1.all()   # [(1, 'Alice'), (2, 'Bob')]
            # 方式2：获取字典列表
            dicts1 = result1.mappings().all()   # [{'id': 1, 'name': 'Alice'}, ...]
            result1.scalars().all()  # ❌ 只返回 id，丢掉 name

            # 查询2：选择整个实体
            stmt2 = select(userPO)
            result2 = await session.execute(stmt2)
            # 方式1：获取实体对象列表
            users = result2.scalars().all()   # [userPO(id=1, name='Alice'), ...]
            # 方式2：获取字典列表（注意：这里每个字典的键是列名，值是列的值，但不会是实体对象）
            dicts2 = result2.mappings().all()   # [{'id': 1, 'name': 'Alice', 'guid': ...}, ...]
        """
        self.session=session 
        pass

    __po_has_field__: str = "_sa_instance_state" 
    @staticmethod
    def remove_db_instance_state(poInstance_or_poList: Iterator | Any) -> List[Dict] | Dict:
        if hasattr(poInstance_or_poList, "__iter__") and not isinstance(poInstance_or_poList, str):
            
            result = [dict(filter(lambda k: k[0] != Actuator.__po_has_field__, a1.__dict__.items())) for a1 in poInstance_or_poList] 
            for r in result:
                for r1 in r:
                    value = r.get(r1)
                    if (hasattr(value, Actuator.__po_has_field__)):    #if (isinstance(value, BasePO)): 
                        dic = Actuator.remove_db_instance_state(value)
                        r.update({r1: dic})
            return result
        # and hasattr (poInstance_or_poList,"__dict__")
        elif hasattr(poInstance_or_poList, "__dict__"):
            return dict(filter(lambda k: k[0] != Actuator.__po_has_field__, poInstance_or_poList.__dict__.items()))
        else:
            return poInstance_or_poList

    @staticmethod
    def row2dict(row: Row) -> Dict:
        """
        xxxxPO.id.label("xxx_id") 为数据取别名
        出现重名覆盖
        """
        d: dict = {}
        for i in range(0, len(row)):
            val = row[i]
            key = row._fields[i]
            if hasattr(val, Actuator.__po_has_field__):
                dc = Actuator.remove_db_instance_state(val) 
                val=dc
            d.update({key: val})
        return d

    @staticmethod
    def one2Dict(fetchone: Row | RowMapping|dict) -> Dict:
        """
        Row:        execute.fetchmany() | execute.fetchone()
        RowMapping: execute.mappings().fetchall()|execute.mappings().fetchone()  
        """
        if isinstance(fetchone, Row):
            return dict(zip(fetchone._fields, fetchone))
        elif isinstance(fetchone, RowMapping):
            return dict(fetchone)
        elif isinstance(fetchone, dict):
            return fetchone
        log.warn(f"未知类型：‘{type(fetchone)}’,直接返回")
        return fetchone

    @staticmethod
    def list2Dict(list: List[Row | RowMapping]) -> List[dict]:
        return [Actuator.one2Dict(a) for a in list]
    @staticmethod
    def select_result_strategy(stmt: Select):
        """
        根据 Select 类型返回推荐的取值方式
        """
        for col in stmt._raw_columns:
            if isinstance(col, Mapper):
                return "scalars" 
        # 单列
        if len(stmt._raw_columns) == 1:
            return "scalar" 
        # 多列
        return "mappings"
    @staticmethod
    def is_entity_select(stmt):
        """判断 Select 是否查询 ORM 实体"""
        for col in stmt._raw_columns:
            if isinstance(col, Mapper):
                return True
        return False

    async def _execute(self, select: Select|Update| Delete| Insert, params: Dict[str, Any] = None):
         exec: ChunkedIteratorResult|CursorResult = await self.session.execute(select, params) 
         return exec 
    
    async def execSQL(self, sql: Update | Insert | Delete, params: Dict | List | Tuple = None):
        """
        执行简单SQL语句
        SQLAlchemy 2.0 中:
            在异步上下文中，AsyncSession.execute()->ChunkedIteratorResult
            在同步上下文中，Session.execute()-> CursorResult
        """
        data: CursorResult = await self.session.execute(sql, params) 
        return  data.rowcount 

    async def execute(self, select: Select|Update| Delete| Insert, params: Dict[str, Any] = None):
        """
        执行sql
        
        返回影响的行数，或者一个对象 
        @example
        execute(Select(po.id,po.name).where(po.id==1))->id|None 
        execute(Select(po).where(po.id==1))->po|None
        execute(Update(po).where(po.id==1).values(name="new"))->影响的行数
        """
        exec = await self._execute(select, params)
        if isinstance(exec, CursorResult):
            return exec.rowcount
        return exec.scalar() 
    async def count(self,  *filters: ColumnElement[bool], column: InstrumentedAttribute = "*") -> int:
        """
        count
        """
        # todo 本来使用的为小写 select
        sql = Select(func.count(column)).filter(and_(*filters))
        return await self.execute(sql)

    async def exist(self,  *filters: ColumnElement[bool], column: InstrumentedAttribute = "*") -> bool:
        """
        是否操作
        """
        count = await self .count(  *filters, column=column)
        return count > 0 
    async def query_one_mappings(self, select: Select,   params: Dict | Tuple | List = None):
        """ 
        select:Select  
        return dict
        """ 
        executer = await self. _execute(select, params) 
        result = executer.mappings().fetchone()
        return Actuator.one2Dict(result)
         
    async def query_all_mappings(self, select: Select,   params: Dict | Tuple | List = None):
        """ 
        select:Select  
        return List[dict]
        """ 
        executer = await self. _execute(select, params) 
        result = executer.mappings().all()
        return Actuator.list2Dict(result)

    async def query_all_entity(self, select: Select,  params: Dict | List | Tuple = None):
        """
        session: AsyncSession
        select:Select 
        return list 
        """ 
        executer  = await   self. _execute(select, params) 
        return executer.scalars().fetchall()
    async def query_all_entity_mappings(self, select: Select,  params: Dict | List | Tuple = None):
        retult=await self.query_all_entity(select, params)
        return Actuator.remove_db_instance_state(retult)

    async def query_one_entity(self, select: Select,params: Dict | List | Tuple = None):
        """
        session: AsyncSession
        select:Select
        remove_db_instance_state: bool 
        return PO|None
        """ 
        executer  = await   self. _execute(select, params) 
        # user: UserPO = executer.scalar()
        data = executer.fetchone() # 返回的是元组
        one = None
        if data is not None:
            one = data[0] 
        return one
    async def query_one_entity_mapping(self, select: Select,params: Dict | List | Tuple = None):
        result= await self.query_one_entity(select, params)
        if result is None:
            return None
        return Actuator.remove_db_instance_state(result)


    def add_all(self, *pos: BasePO):
        self.session.add_all(pos)
        
    async def query_tree(self,select: Select, rootValue: any = None, pid_field: str = "pid", id_field: str = "id",   param: Dict | List | Tuple = None):
        """
        执行查询: tree列表 
        """
        try:
            if self.is_entity_select(select):
                result = await self.query_all_entity_mappings(select,  param)
            else:
                result = await self.query_all_mappings(select,  param)
            if result is None:
                treeList = []
            else:
                treeList = list_to_tree(result, rootValue, pid_field, id_field)
            if len(treeList) == 0:
                return  []
            return treeList
        except Exception as e:
            log.err(f"查询失败{e}",e) 

    async def query_page(self, filter:absFilterItems):
        """
        分页查询
        """ 
        try:
            #filter.count_select, filter.list_select 
            total  = await self.execute(filter.count_select) 
            if self.is_entity_select( filter.list_select ): 
                result=self.query_all_entity_mappings(filter.list_select ) 
            else:
                result=await self.query_all_entity_mappings(filter.list_select) 
            return total, result 
        except Exception as e:
            log.err(f"查询失败{e}",e)
            return None, None

     
    async def batchAdd(self,  poList: List[BasePO],   userId=None, checkFun:Callable[[POType,AsyncSession], Any]  =None, afterFun: Callable[[List[BasePO], AsyncSession,Any], Result] = None):
        try: 
            for po in poList:
                po.add_assignment(userId)
                result=None
                if checkFun is not None:
                    result = await checkFun(po,self. session)
                    
            self. add_all(poList)
            if afterFun is not None:
                self.session.flush()
                await afterFun(poList, self.session,result)
            return True

           
        except Exception as e:
            log.err(f"批量增加失败{e}",e)
            return None

     
    async def add(self,  option:OperationOption):
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
            if option.poType is None and option.po is None:
                return Result.fail(message="实体对象不能为空")
            if option.json is None:
                return Result.fail(message="json 不能为空")
            po=option.po if option.po is not None else option.poType(**option.json) 
            po.add_assignment(option.userId)
            if option.beforeFun is not None:
                result=await option.beforeFun(po, self)
                if result is not None:
                    return result
            self.session.add(po)
            if option.afterFun is not None:
                self.session.flush()
                result=await option.afterFun(po, self)
                if result is not None:
                    return result
            return Result.success()
        except Exception as e:
            log.err(f"执行add err {e}",e)
            return Result.fail(message=f"增加失败{e}")
            

    async def edit(self, option:OperationOption):
        """
        编辑 
        option.select 和 option.po 优先使用 select
        """
        try:
            if option.select is None and option.po is None:
                return Result.fail(message="未能找到实体对象") 
            if option.json is None:
                return Result.fail(message="json 不能为空")
            oldPo:Optional[BasePO] =   await self.query_one_entity( option.select )  if option.select is not None else option.po
            if oldPo is  None:
                return   Result.fail(message=f"未找到‘{option.select}’对应的信息!")
            oldPo.edit_assignment(option.userId)
            newPO=option.poType(**option.json)
            if option.json is not None: 
                oldPo.update(newPO)
            if option.beforeFun :
                return await option.beforeFun (oldPo, newPO,self ) 
            return  Result.success()
            
        except Exception as e:
            log.err(f"执行edit失败{e}",e)
            return  Result.fail(message=f"编辑失败{e}")

    async def delete(self,po:POType):
        await  self.session.delete(po)  

    async def remove(self,option:OperationOption):
        """
        删除 

        request: Request, 
        pk: any,      #主键值
        poType: TypeVar # 实体类型
        checkFun(oldPo, session),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
                    返回值 Result 返回
                    返回   bool   回滚
                  
        @afterFun afterFun(oldPo, session, await checkFun())    删除成功后执行
        return Result
        """ 
        try: 
            oldPo: Optional[BasePO] =None
            while True:
                if option.po is not None:
                    oldPo=option.po 
                    break
                if option.select  is not None :
                    oldPo  = await self.query_one_entity( option.select)
                    break 
                if option.pk  is not None and option.poType is not None: 
                    oldPo  = await self. session.get_one(option.poType, option.pk) 
                    break 
                break
            if oldPo is None:
                return Result.fail(message="未能找到实体对象或主键值") 
            result=None
            if option.beforeFun is not None:
                result = await option.beforeFun(oldPo, self) 
            if isinstance(result, Result):
                return result
            elif isinstance(result, bool) and result:
                await  self.delete(oldPo)
                self.session.flush()
                if option.afterFun is not None:
                    return await option.afterFun(oldPo, self )
            else:
                return Result.fail(result,"删除失败beforeFun 返回结果类型bool,且为False") 
            return Result.success()
        except Exception as e:
            await self.session.rollback()
            return self._error0(e)
            
      
    async def operation(self, option:OperationOption):
        """
        执行操作
        """ 
        if option.type == OperationType.INSERT:
            return await self.add(option) 
        if option.type == OperationType.UPDATE:
            return await self.edit(option) 
        if option.type == OperationType.DELETE:
            return await self.remove(option) 
        if option.type == OperationType.QUERY:
            return await self.query(option) 
        
	
class OperationOption:
    def __init__(self):
        self.type:OperationType = OperationType.INSERT
        self.json: Optional[Dict]=None
        self.filter: Optional[absFilterItems]=None
        self.poType:  Optional[POType]=None
        self.po: Optional[ BasePO]=None
        self.userId: Optional[int]=None
        self.beforeFun:Callable[[BasePO, Actuator], Awaitable[Result]]=None
        self.afterFun:Callable[[BasePO, Actuator], Awaitable[Result]]=None
        self.select: Optional[Select]=None
        self.pk: Optional[Any]=None
    @classmethod
    def create_add(cls,json:Dict,*, po:BasePO=None, poType:POType=None,  userId:int=None, beforeFun:Callable[[BasePO, Actuator], Awaitable[Result]]=None, afterFun:Callable[[BasePO, Actuator], Awaitable[Result]]=None):
        option=cls()
        option.type=OperationType.INSERT
        option.poType=poType
        option.po=po
        option.json=json
        option.userId=userId
        option.beforeFun=beforeFun
        option.afterFun=afterFun
        return option
    @classmethod
    def create_edit(cls,json:Dict,poType:POType,*,select:Select=None,  po:BasePO=None,   userId:int=None, beforeFun:Callable[[BasePO, Actuator], Awaitable[Result]]=None):
        option=cls()
        option.type=OperationType.UPDATE
        option.poType=poType
        option.po=po
        option.select=select
        option.json=json
        option.userId=userId
        option.beforeFun=beforeFun
        return option
    @classmethod
    def create_del(cls,*,select:Select=None,pk:Any=None,po:BasePO=None, poType:POType=None,checkFun:Callable[[BasePO,Actuator],Optional[bool|Result]]  =None, afterFun:Callable[[BasePO,Actuator], Result]=None):
        option=cls()
        option.type=OperationType.DELETE
        option.pk=pk

        option.poType=poType
        option.select=select

        option.po=po
        option.beforeFun=checkFun
        option.afterFun=afterFun
        return option 

