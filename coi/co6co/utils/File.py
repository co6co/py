# -*- coding:utf-8 -*-

import json,base64,os

class File:

    @staticmethod
    def readJsonFile(filePath:str, encoding:str="utf-8"):
        """
        filePath: 文件路径
        encoding:文件编码
        return     json 对象
        """
        with open(filePath, "r", encoding=encoding) as f:
            result = json.load(f) 
            return result
    
    @staticmethod
    def writeJsonFile(filePath,obj):
        """
        filePath: 文件路径
        obj : 待写入的对象
        """
        updated_list = json.dumps( obj, sort_keys=False, indent=2, ensure_ascii=False)
        with open(filePath, 'w', encoding='utf-8') as file:
            file.write(updated_list) 

    @staticmethod
    def readFile(filePath: str, encoding: str = "utf-8") -> str:
        """
        filePath: 文件路径
        encoding:文件编码
        return content 
        """
        with open(filePath,"r",encoding=encoding) as file: 
            content=file.read() #.splitlines()# readlines() 会存在\n 
            return content
    @staticmethod
    def createFolder(filePath:str,isFilePath:False)->str:
        """
        创建路径,并返回 传入的路径 以当前操作系统 分隔路径符号分割
        """
        filePath=os.path.abspath(filePath) # 转换为 os 所在系统路径 
        folder=filePath
        if isFilePath: folder=os.path.dirname(filePath) 
        if not os.path.exists(folder):os.makedirs(folder)
        return filePath
    
    @staticmethod
    def writeFile(filePath,data:bytes):
        """
        filePath: 文件路径
        obj : 待写入的对象
        """ 
        filePath=File.createFolder(filePath,True)
        with open(filePath,"wb") as file: #二进制模式打开文件
            file.write(data)
    @staticmethod
    def writeBase64ToFile(filePath,base64Content:str):
        """
        filePath: 文件路径
        obj : 待写入的对象
        """
        data=base64.b64decode(base64Content)
        File.writeFile(filePath,data)
            