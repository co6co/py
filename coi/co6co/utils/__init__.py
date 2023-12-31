#-*- coding:utf-8 -*-
import re,random,string,time,datetime

def isBase64(content:str)->bool:
    _reg="^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$"
    group=re.match(_reg,content)
    if group !=None:return True
    return False

def getRandomStr(length:int,scope:str=string.ascii_letters+string.digits)->str:
    return ''.join(random.sample(scope,length))

def generate_id(*_):
    return time.time_ns()
def getDateFolder(format:str="%Y/%m/%d"):
    """
    获得当前日期目录:
    2023/12/01
    """
    time = datetime.datetime.now() 
    return f"{time.strftime(format)}"
