from enum import Enum,unique
# from abc import ABC, abstractclassmethod  抽象
from typing import List,Dict

@unique #帮助检查 保证没有重复值
class Base_Enum(Enum): 
    """
    枚举[名字 数字]
    """
    def __new__(cls, english:str, chinese:str, value:int):
        obj = object.__new__(cls)
        obj.name = english 
        obj.val = value # value 为元组 (en_name,cn_name,val)
        return obj 
    
    @classmethod
    def to_dict_list(cls)->List[Dict]: 
        status=[{ 'name':i.name,'value':i.val} for i in cls]
        return status 
    def getValue(self)->int:
        return self.val
    def getName(self)->int:
        return self.name 

@unique
class Base_EC_Enum(Enum):
    """
    枚举[英文 中文 数字] 
    """
    def __new__(cls, english:str, chinese:str, value:int):
        obj = object.__new__(cls)
        obj.en_name = english
        obj.cn_name = chinese
        obj.val = value # value 为元组 (en_name,cn_name,val)
        return obj 
    
    @classmethod
    def to_dict_list(cls)->List[Dict]: 
        status=[{"key":i.en_name,'label':i.cn_name,'value':i.val} for i in cls]
        return status 
    def getValue(self)->int:
        return self.val
    def getName(self)->int:
        return self.en_name
    def getCnName(self)->int:
        return self.cn_name
 
    