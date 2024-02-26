

from typing import TypeVar,Tuple,List,Dict,Any,Union,Iterator
from sqlalchemy.engine.row import  Row
from .po import BasePO
from sqlalchemy.sql import Select

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

    def mapping(executeResult:any)-> List[dict]:  
        #sqlalchemy.engine.result.ChunkedIteratorResult 
        return [dict(zip(a._fields,a))  for a in  executeResult]