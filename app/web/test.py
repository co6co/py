
from typing import Tuple,ValuesView,TypeVar,List

def test(a:int ,b:str,*param:str, c:bool=True,**kv:bool):
    print(c)
    print(a,b,c,*param,*kv)
     

test(1,"b",'d',"d",k=321,g=546)
    