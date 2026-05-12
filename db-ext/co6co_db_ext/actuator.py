from .po import BasePO 
from .db_filter import absFilterItems

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Select, Column, Integer, String, Update, Delete, Insert
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.row import Row, RowMapping
from sqlalchemy.orm import Mapper

from typing import Callable, Any, Dict , List ,Tuple
from functools import wraps
 
from co6co.data.result import Result,Page_Result
from co6co.utils import log 
from co6co.utils.tool_util import list_to_tree 
  
from sqlalchemy import func, text,and_
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute 

from typing import TypeVar, Iterator,Dict, List, Any, Tuple, Optional, Callable

 


# 定义泛型：限定必须是 BasePO 的子类
POType = TypeVar("POType", bound=BasePO) 

class Actuator:
    
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
        if hasattr(poInstance_or_poList, "__iter__"):
            result = [dict(filter(lambda k: k[0] != Actuator.__po_has_field__,
                           a1.__dict__.items())) for a1 in poInstance_or_poList]
            for r in result:
                for r1 in r:
                    value = r.get(r1)
                    if (isinstance(value, BasePO)):
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
            c = row[i]
            if hasattr(c, Actuator.__po_has_field__):
                dc = Actuator.remove_db_instance_state(c)
                d.update(dc)
            else:
                key = row._fields[i]
                '''
                j=1 
                while key in d.keys():
                    key=f"{ row._fields[i]}_{str(j )}"
                    j+=1
                '''
                d.update({key: c})
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
    def is_entity_select(stmt):
        """判断 Select 是否查询 ORM 实体"""
        for col in stmt._raw_columns:
            if isinstance(col, Mapper):
                return True
        return False

    async def _execute(self, select: Select|Update| Delete| Insert, params: Dict[str, Any] = None):
         exec: ChunkedIteratorResult = await self.session.execute(select, params) 
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
        """
        exec = await self._execute(select, params)
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
            filter.count_select, filter.list_select 
            total =   await self.execute(filter.count_select ) 
            if self.is_entity_select( filter.list_select ): 
                result=self.query_all_entity_mappings(filter.list_select ) 
            else:
                result=await self.query_all_entity_mappings(filter.list_select) 
            return total, result 
        except Exception as e:
            log.err(f"查询失败{e}",e)
            return None

     
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

    async def add(self,  po: BasePO,  userId=None,  checkFun:Callable[[POType,AsyncSession], Any]  =None, afterFun: Callable[[BasePO, AsyncSession,Any], Result] = None):
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
            po.add_assignment(userId)
            result=None
            if checkFun is not None:
                result = await checkFun(po, self. session ) 
            self.session.add(po)
            result=Result.success()
            if afterFun is not  None:
                self.session.flush()
                result= await afterFun(po, self.session,result )
            return result 
        except Exception as e:
            log.err(f"执行add err {e}",e)
            return Result.fail(message=f"增加失败{e}")
            

    async def edit(self,  select:Select,  newPO: POType , userId=None, fun:Callable[[POType,POType, AsyncSession],Result]=None, json2Po: bool = True):
        """
        编辑 
        pk: any,          # 主键
        poType: TypeVar,  # 实体类型
        newPO:BasePO    ,    #  newPo 更新到 oldPO
        userId=None, # 用户ID
        fun=None,    # 执行一些其他操作， (oldPO,po,session)
                     # 返回值将直接返回客户端并且回滚数据库操作
        json2Po:bool # 根据 请求的json 转换的对象更新 实体对象，在 fun 之前执行

        return JSONResponse
        """
        try:  
            oldPo:BasePO = await self.query_one_entity( select ) 
            if oldPo is  None:
                return   Result.fail(message=f"未找到‘{select}’对应的信息!")
            oldPo.edit_assignment(userId)
            if json2Po:
                oldPo.update(newPO)
            if fun :
                return await fun(oldPo, newPO,self. session ) 
            return  Result.success()
            
        except Exception as e:
            log.err(f"执行edit失败{e}",e)
    async def delete(self,po:POType):
        await  self.session.delete(po) 
        pass

    async def remove(self,  pk: any, poType: POType,  checkFun:Callable[[POType,AsyncSession],Optional[bool|Result]]  =None  , afterFun:Callable[[POType,AsyncSession], Result]=None):
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
            oldPo: BasePO = await self. session.get_one(poType, pk)
            if oldPo is None:
                return Result.fail(message=f"未找到‘{pk}’对应的信息!")
            result=None
            if checkFun is not None:
                result = await checkFun(oldPo, self.session) 
            if isinstance(result, Result):
                return result
            elif isinstance(result, bool) and result:
                await  self.delete(oldPo)  
                if afterFun is not None:
                    return await afterFun(oldPo, self.session )
            else:
                return Result.fail(result,f"删除失败:checkFun 返回结果类型{type( checkFun)},{result}") 
            return Result.success()
        except Exception as e:
            await self.session.rollback()
            log.err(f"执行remove失败{e}",e)
            

	
