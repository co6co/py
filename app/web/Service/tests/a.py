from enum import Enum, unique
from typing import TypeVar, List, Type,Optional
T = TypeVar('T')
E = TypeVar('E', bound='Base_Enum')

class Base_Enum (Enum): 
    key: T 
    val: T 
    def __new__(cls, key: T, value: T):
        _value = len(cls.__members__) + 1  # 为每个成员分配一个递增的整数值
        obj = object.__new__(cls)
        obj.key = key
        obj.val = value  # value 为元组 (en_name,cn_name,val)
        obj._value_ = _value  # 设置枚举成员的值
        return obj
    @classmethod 
    def key2enum(cls: Type[E], key)-> Optional[E]: 
        for i in cls: 
            if i.key == key:  
                return i   
        return None 
class DeviceVender(Base_Enum): 
    Uniview = 'UNIVIEW',  1 

t=DeviceVender.key2enum('UNIVIEW')
print(t,type(t))
