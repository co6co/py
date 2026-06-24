

from typing import   Tuple, List, Dict, Any,   Iterator, Callable
from sqlalchemy.engine.row import Row, RowMapping
from .po import BasePO
from co6co.utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Select, Update, Insert, Delete  
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute 

from .db_filter import absFilterItems
from .actuator import Actuator
from .session import session_context

class db_tools:
    """
    数据转换工具
    1. 
    data=  exec.mappings().all() 
    result=[dict(zip( a.keys(),a._to_tuple_instance())) for a in  data] 

    2.
    [dict(zip(a._fields,a))  for a in  executeResult]
    """
    @staticmethod
    def remove_db_instance_state(poInstance_or_poList: Iterator | Any) -> List[Dict] | Dict: 
        return Actuator.remove_db_instance_state(poInstance_or_poList)

    @staticmethod
    def row2dict(row: Row) -> Dict: 
        return Actuator.row2dict(row)

    @staticmethod
    def one2Dict(fetchone: Row | RowMapping|dict) -> Dict:
        """
        Row:        execute.fetchmany() | execute.fetchone()
        RowMapping: execute.mappings().fetchall()|execute.mappings().fetchone()  
        """
        return Actuator.one2Dict(fetchone)

    @staticmethod
    def list2Dict(list: List[Row | RowMapping]) -> List[dict]: 
        return Actuator.list2Dict(list)
        
    @staticmethod
    def _get_actuator(session: AsyncSession|Actuator) -> Actuator:
        if isinstance(session, AsyncSession):
            actuator= Actuator(session )
        elif isinstance(session, Actuator):
            actuator =session
        else:
            raise ValueError("session must be AsyncSession or Actuator")
        return actuator

    '''
    def mapping(executeResult: any) -> List[dict]:
        """
        不在使用
        """
        # sqlalchemy.engine.result.ChunkedIteratorResult
        return [dict(zip(a._fields, a)) for a in executeResult]
    '''
    async def execSelect(session: AsyncSession|Actuator, select: Select, params: Dict | List | Tuple = None) -> int | None:
        """
        执行查询语句
        @return: int | None
        """
        actuator=db_tools._get_actuator(session)
        return await actuator.execute(select, params) 

    async def count(session: AsyncSession|Actuator, *filters: ColumnElement[bool], column: InstrumentedAttribute = "*") -> int:
        """
        count
        """
        actuator=db_tools._get_actuator(session)
        return await actuator.count( *filters,column=column) 

    async def exist(session: AsyncSession|Actuator, *filters: ColumnElement[bool], column: InstrumentedAttribute = "*") -> bool:
        """
        exist
        """
        actuator=db_tools._get_actuator(session)
        return await actuator.exist( *filters,column=column) 

    async def execForMappings(session: AsyncSession|Actuator, select: Select, queryOne: bool = False, params: Dict | Tuple | List = None):
        """
        session: AsyncSession
        select:Select 

        return list
        """ 
        actuator=db_tools._get_actuator(session)
        if queryOne:
            return await actuator.query_one_mappings(select, params) 
        return await actuator.query_all_mappings(select, params)  

    async def execForPos(session: AsyncSession|Actuator, select: Select, remove_db_instance_state: bool = True, params: Dict | List | Tuple = None):
        """
        session: AsyncSession
        select:Select
        remove_db_instance_state: bool

        return list
        """
        actuator=db_tools._get_actuator(session)
        if remove_db_instance_state:
            return await actuator.query_all_entity_mappings(select, params) 
        return await actuator.query_all_entity(select, params)   

    async def execForPo(session: AsyncSession|Actuator, select: Select, remove_db_instance_state: bool = True, params: Dict | List | Tuple = None):
        """
        session: AsyncSession
        select:Select
        remove_db_instance_state: bool

        return PO|None
        """
        actuator=db_tools._get_actuator(session)
        if remove_db_instance_state:
            return await actuator.query_one_entity_mapping(select, params) 
        return await actuator.query_one_entity(select, params)   
         

    async def execSQL(session: AsyncSession|Actuator, sql: Update | Insert | Delete, sqlParam: Dict | List | Tuple = None):
        """
        执行简单SQL语句
        """
        actuator=db_tools._get_actuator(session)
        return  await actuator.execSQL(sql, sqlParam) 


'''
exec.fetchone() //None| (data,)
exec.mappings().fetchone()  // {'id': 1, 'userName': 'admin'} | {"userPO":PO}
exec..fetchone()    //(1, 'admin') || po
'''


class DbCallable:
    session: AsyncSession = None

    def __init__(self, session: AsyncSession):
        self.session = session

    async def __call__(self, func: Callable[[Actuator], Any], *args, **kwargs):
        """
        with self.session, self.session.begin():
            这会创建一个显式的事务块
            在 with 块内的所有操作会在同一个事务中执行
            当代码块正常执行完毕，事务会自动提交
            如果发生异常，事务会自动回滚
            这是 SQLAlchemy 推荐的显式事务处理方式

        with self.session:
            仅表示获取会话的上下文管理
            不会自动创建事务，除非在会话配置中设置了 autocommit=False(默认值)
            在这种模式下，需要手动调用 session.commit() 或 session.rollback()
            如果没有显式提交，会话关闭时可能会导致事务回滚
        """
        async with session_context(self.session)() as session:
        #async with self.session, self.session.begin():
            if func is not None: 
                try:
                    actuator= Actuator(session)
                    return await func(actuator,*args, **kwargs)
                except Exception as e:
                    #await actuator.session.rollback() # 是否需要回滚
                    log.err(f"执行'DBCallable'异常:{e}",e)
                    raise


class QueryOneCallable(DbCallable):
    async def __call__(self, select: Select, isPO: bool = True, param: Dict | List | Tuple = None):
        async def exec(actuator: Actuator): 
            if isPO:
                return await actuator.query_one_entity(select, param) 
            else:
                return await actuator.query_one_mappings(select, param)  
        return await super().__call__(exec)


class InsertCallable(DbCallable):
    async def __call__(self, *po: BasePO):
        async def exec(actuator: Actuator):
             actuator.add_all(*po)  
        return await super().__call__(exec)


class UpdateOneCallable(DbCallable):
    async def __call__(self, queryOneSelect: Select, editFn: Callable[[AsyncSession, Any], None | Any] = None, param: Dict | List | Tuple = None):
        """
        queryOneSelect: 查询语句
        editFn: (session,po)->Any|None   返回:None  ->回滚,
                                            :Any   -> 函数返回值
        """
        async def exec(actuator: Actuator): 
            one = await actuator.query_one_entity(queryOneSelect, param)
            if editFn is not None:
                result = await editFn( actuator.session, one)
                if result is None:
                    await actuator.session.rollback()
                return result

        return await super().__call__(exec)


class QueryListCallable(DbCallable):
    async def __call__(self, select: Select, isPO: bool = True, remove_db_instance=True, param: Dict | List | Tuple = None):
        async def exec(actuator: Actuator):
            if isPO:
                result=await actuator.query_all_entity_mappings(select,param) if remove_db_instance else await actuator.query_all_entity(select, param)
            else: 
                result=await actuator.query_all_mappings(select,param)
            return result
        # return await super(QueryListCallable,self).__call__(exec) #// 2.x 写法
        return await super().__call__(exec)


class QueryPagedCallable(DbCallable):
    async def __call__(self, countSelect: Select, select: Select, isPO: bool = True, remove_db_instance=True, param: Dict | List | Tuple = None) -> Tuple[int, List[dict]]:
        async def exec(actuator: Actuator):
            total =   await actuator.execute(countSelect, param) 
            if isPO:
                result=await actuator.query_all_entity_mappings(select,param) if remove_db_instance else await actuator.query_all_entity(select, param)
            else:
                result=await actuator.query_all_mappings(select,param) 
            return total, result
        return await super().__call__(exec)


class QueryPagedByFilterCallable(QueryPagedCallable):
    async def __call__(self, filter: absFilterItems, isPO: bool = True, remove_db_instance=True, param: Dict | List | Tuple = None) -> Tuple[int, List[dict]]:
        return await super().__call__(filter.count_select, filter.list_select, isPO, remove_db_instance, param)
