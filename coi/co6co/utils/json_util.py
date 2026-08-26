# -*- coding:utf-8 -*-


from __future__ import annotations
import json
import datetime
from typing import Any


class JSONEncoder(json.JSONEncoder): 
    _instance = None 
    def __new__(cls, *args, **kwargs): 
        if cls._instance is None:  
            #debug 类实例依赖第一次使用的人，如果第一次使用的人传入的参数比较特别，类实例可以会有特殊的行为
            cls._instance = super().__new__(cls)
            cls._instance._initialized=False
            return cls._instance
        else: 
            return super().__new__(cls) 

    @property
    def instance(self):
        # 好像没什么用 已经实例了，为什么还需要全局的实例
        #先查找实例的__dict__，如果没有找到，就会去类里找
        return self._instance
    
    def __init__(self, ensure_ascii=False,  **kwargs):
        if hasattr(self, '_initialized') and not self._initialized and self._instance==self:
            self._initialized = True  
        super().__init__(ensure_ascii=ensure_ascii, **kwargs)   

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, tuple):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return None
        
    @staticmethod
    def loads(s: str | bytes | bytearray|Any, jsonEncoder: type[json.JSONDecoder] | None = None):
        """
        解析JSON字符串
        :param s: JSON字符串 或者类
        :param jsonEncoder: 解析器
        :return: dict
        """
        if not isinstance(s, str) and not isinstance(s, bytes) and not isinstance(s, bytearray):
            s = JSONEncoder.dumps(s)
            return json.loads(s,cls=jsonEncoder)  
        return json.loads(s,cls=jsonEncoder)

    @staticmethod
    def get_instance():
         if  JSONEncoder._instance is None: 
                JSONEncoder._instance = JSONEncoder( )  
         return JSONEncoder._instance
    @staticmethod
    def dumps(res: any, jsonEncoder: json.JSONEncoder = None):
        if jsonEncoder == None: 
            return  JSONEncoder.get_instance().encode(res)
        else:
            return  jsonEncoder.encode(res)
