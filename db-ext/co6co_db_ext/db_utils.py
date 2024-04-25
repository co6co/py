

from typing import TypeVar,Tuple,List,Dict,Any,Union,Iterator
from sqlalchemy.engine.row import  Row,RowMapping
from .po import BasePO 
from co6co.utils import log

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Select
from .db_filter import absFilterItems
from  sqlalchemy.engine.result import ChunkedIteratorResult

class db_tools:
    """
    数据转换工具
    1. 
    data=  exec.mappings().all() 
    result=[dict(zip( a.keys(),a._to_tuple_instance())) for a in  data] 
    
    2.
    [dict(zip(a._fields,a))  for a in  executeResult]
    """ 
    __po_has_field__:str="_sa_instance_state" 
    @staticmethod
    def remove_db_instance_state( poInstance_or_poList:Iterator| Any )->List[Dict]|Dict:
        if  hasattr (poInstance_or_poList,"__iter__") :
            result= [dict(filter(lambda k: k[0] !=db_tools.__po_has_field__, a1.__dict__.items())) for a1 in poInstance_or_poList]  
            for r in result:
                for r1 in r:
                    value=r.get(r1)
                    if (isinstance(value,BasePO)):
                        dic=db_tools.remove_db_instance_state(value)
                        r.update({r1:dic}) 
            return result
        #and hasattr (poInstance_or_poList,"__dict__")
        elif hasattr (poInstance_or_poList,"__dict__"): return dict(filter(lambda k: k[0] !=db_tools.__po_has_field__, poInstance_or_poList.__dict__.items()))
        else: return poInstance_or_poList
        
    @staticmethod
    def row2dict(row:Row)->Dict:
        """
        xxxxPO.id.label("xxx_id") 为数据取别名
        出现重名覆盖
        """
        d:dict={}
        for i in range(0,len(row)):
            c=row[i]
            if hasattr(c,db_tools.__po_has_field__):
                dc=db_tools.remove_db_instance_state(c)
                d.update(dc)
            else:
                key=row._fields[i]
                '''
                j=1 
                while key in d.keys():
                    key=f"{ row._fields[i]}_{str(j )}"
                    j+=1
                '''
                d.update({key:c})
        return d
    @staticmethod
    def one2Dict(fetchone:Row|RowMapping)->Dict:
        """
        Row:        execute.fetchmany() | execute.fetchone()
        RowMapping: execute.mappings().fetchall()|execute.mappings().fetchone()  
        """
        if type(fetchone)==Row: return dict(zip(fetchone._fields,fetchone)) 
        elif  type(fetchone)==RowMapping: return dict(fetchone)
        elif  type(fetchone)==dict: return fetchone
        log.warn(f"未知类型：‘{type(fetchone)}’,直接返回")
        return fetchone
    
    @staticmethod
    def list2Dict(list:List[Row|RowMapping])->List[dict]: 
        return [db_tools.one2Dict(a)  for a in  list]
    
    def mapping(executeResult:any)-> List[dict]: 
        """
        不在使用
        """ 
        #sqlalchemy.engine.result.ChunkedIteratorResult 
        return [dict(zip(a._fields,a))  for a in  executeResult]
    
    async def execForMappings(session:AsyncSession,select:Select):
        """
        session: AsyncSession
        select:Select 

        return list
        """ 
        exec=await session.execute(select)  
        data=  exec.mappings().all()  
        result=db_tools.list2Dict(data)  
        return result
    
    async def execForPos(session:AsyncSession,select:Select, remove_db_instance_state:bool=True):
        """
        session: AsyncSession
        select:Select
        remove_db_instance_state: bool

        return list
        """

        exec:ChunkedIteratorResult=await session.execute(select)
        if remove_db_instance_state:
            return db_tools.remove_db_instance_state(exec.scalars().fetchall())  
        else:
            return exec.scalars().fetchall() 

    

'''
exec.fetchone() //None| (data,)
exec.mappings().fetchone()  // {'id': 1, 'userName': 'admin'} | {"userPO":PO}
exec..fetchone()    //(1, 'admin') || po
'''

class DbCallable:
    session:AsyncSession=None
    def __init__(self,session:AsyncSession):
        self.session=session
    async def __call__(self, func ):
        async with self.session,self.session.begin():
            if func!=None:return await func(self.session)
            
class QueryOneCallable(DbCallable):  
    async def __call__(self, select :Select,isPO:bool=True):
        async def exec(session:AsyncSession):
            exec=await session.execute(select)  
            if isPO:  
                data=exec.fetchone()
                # 返回的是元组
                if data!=None:return data[0]
                else: return None
            else:
                data=exec.mappings().fetchone()
                result=db_tools.one2Dict(data)  
                return result 
        return await super().__call__(exec)
         
class QueryListCallable(DbCallable):  
    async def __call__(self, select :Select,isPO:bool=True,remove_db_instance=True):
        async def exec(session:AsyncSession):
            if isPO: 
               result=await db_tools.execForPos(session,select,remove_db_instance)
            else:
               result=await db_tools.execForMappings(session,select)
            return result
        #return await super(QueryListCallable,self).__call__(exec) #// 2.x 写法
        return await super().__call__(exec)

class QueryPagedCallable(DbCallable):  
    async def __call__(self, countSelect:Select, select :Select,isPO:bool=True,remove_db_instance=True) ->Tuple[int,List[dict]]:
        async def exec(session:AsyncSession):
            total=await session.execute(countSelect)
            total=total.scalar()  
            if isPO: 
                result=await db_tools.execForPos(session,select,remove_db_instance)
            else: 
                result=await db_tools.execForMappings(session,select) 

            return total,result  
        return await super().__call__(exec)

class QueryPagedByFilterCallable(QueryPagedCallable):  
    async def __call__(self, filter:absFilterItems,isPO:bool=True,remove_db_instance=True) ->Tuple[int,List[dict]]:   
        return await super().__call__(filter.count_select,filter.list_select,isPO ,remove_db_instance) 

