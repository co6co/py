import jwt, datetime
from sanic.request import Request 
from typing import Any
from co6co.utils import log

class JWT_service :
    def __init__(self,secret:str,issuer:str="JWT+SERVICE") -> None:
        self._secret=secret
        self._issuer=issuer
        pass

    def encode(self,data,expiration_date:int=1)->str:
        """
        加密数据： 
        """
        dic={
            'exp':datetime.datetime.utcnow()+datetime.timedelta(days=expiration_date), # 过期时间
            'iat':datetime.datetime.utcnow(), # 开始时间
            'iss':self._issuer,# 签名
            'data':data # 内容一般放用户ID 和开始时间 
        }
        return jwt.encode(dic,self._secret, algorithm="HS256") # 加密字符

    def decode(self,token:str)->Any|None: 
        try:
            return jwt.decode(token,self._secret,issuer=self._issuer,algorithms=['HS256']) # 解密签名  //
        except Exception as e:
            log.err(e)
            return None

async def createToken(request:Request,data:dict):
    """
    登录时生成tooken
    """
    svc=JWT_service(request.app.config.SECRET)
    result=svc.encode(data )   
    return result

async def validToken(request:Request):
    """
    验证token 是否有效
    """
    svc=JWT_service(request.app.config.SECRET)
    log.succ(f"token:{request.token}")
    result=svc.decode(request.token) 

    log.succ(f"tokenparse:{result}")
    if result ==None or 'data' not in  result:return False
    request.ctx.current_user=result["data"]
    log.succ(request.ctx.current_user)
    return True