# -*- encoding:utf-8 -*-
from __future__ import annotations 
class Result:
    def __init__(self) -> None:
        self.code=0
        self.message=""
        self.data={} 
        pass
    
    @staticmethod
    def create(code:int=0,data:any=None,message:str="操作成功")-> Result:
        result=Result()
        result.code=code
        result.data=data
        result.message=message
        return result 
    @staticmethod
    def success(data:any=None,message:str="操作成功")-> Result:
        return Result.create(data=data,code=0, message=message)
    @staticmethod
    def fail(data:any=None,message:str="处理失败")-> Result:
        return Result.create(data=data,code=500, message=message)
 

class Page_Result(Result):
    def __init__(self) -> None:
        super().__init__() 
        self.total=-1

    @staticmethod
    def create(code:int=0,data:any=None,message:str="操作成功")-> Page_Result:
        result=Page_Result()
        result.code=code
        result.data=data
        return result 
    @staticmethod
    def success(data:any=None,message:str="操作成功")-> Page_Result:
         return Result.create(data=data,code=0, message=message)
    @staticmethod
    def fail(data:any=None,message:str="处理失败")-> Page_Result:
        return Result.create(data=data,code=500, message=message)

        

