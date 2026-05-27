import jwt

from datetime import datetime,timezone, timedelta
from typing import Any
from co6co.utils import log
from dataclasses import dataclass

@dataclass(init=False)
class PageToken:
    token: str
    role: str
    expireSeconds:int
    refreshToken: str 

class JwtService:
    """
    设计目标：
    让服务器“无状态地知道你是谁、有什么权限，但数据不能太多否则数据太多影响存储”
    与session不同，session是有状态的，session，客户端只存储ID，服务的需要保存相关数据
    
    1. Refresh Token 绝不能​ 放 localStorage
    2. 必须放在 HttpOnly Cookie 中，
        Set-Cookie: refresh_token=xxxx; HttpOnly; Secure; SameSite=Strict
    3. 常见的刷新token设计方案：
    {
      "uid": 10086,
      "deviceId": "abc123",
      "userAgent": "Chrome/Win",
      "exp": 1700000000
    }
    """
    def __init__(self, secret: str, issuer: str = "JWT+SERVICE") -> None:
        """
        初始化 JWT 服务
        :param secret: 密钥
        :param issuer: 签发者
        """
        self._secret = secret
        self._issuer = issuer
        pass

    def encode(self, data: dict, expire_seconds: int = 86400) -> str:
        """
        加密数据： 
        :param data: 要加密的数据
        :param expire_seconds: 过期时间

        :return: 加密后的字符串
        :except: 加密失败
        """
        now=datetime.now(timezone.utc)
        dic = {
            'exp': now+timedelta(seconds=expire_seconds),  # 过期时间
            'iat': now,  # 签发时间
            'iss': self._issuer,  # 签发者
            'data': data  # 内容一般放用户ID 和开始时间
        }
        return jwt.encode(dic, self._secret, algorithm="HS256")  # 加密字符

    def decode(self, token: str) -> Any | None:
        try:
            if token is None or token == "":
                log.warn("token is None!")
                return None
            # 解密签名  //
            return jwt.decode(token, self._secret, issuer=self._issuer, algorithms=['HS256'])
        except Exception as e:
            log.warn("校验失败:", e)
            return None
    def create_token(self, data: dict,refreshData:dict, expire_seconds: int = 86400, k:int=3, **kvarg):
        """
        kvarg: 其他参数,直接以明文方式增加到字典中
        生成登录token和刷新token
        """
        token = self.encode(data, expire_seconds)
        token2 = self.encode(refreshData, k*expire_seconds)
        result={}
        accessToken=  {"token": token, "expireSeconds": expire_seconds} 
        refreshToken= {"token": token2, "expireSeconds": k*expire_seconds } 
        result.update(kvarg)
        
        result.update({"accessToken":accessToken, "refreshToken":  refreshToken})
        return result


async def createToken(SECRET: str, data: dict, expire_seconds: int = 86400):
    """
    登录时生成tooken
    :param SECRET: 密钥
    :param data: 要加密的数据
    :param expire_seconds: 过期时间
    :return: 加密后的字符串
    :except: 加密失败
    """
    svc = JwtService(SECRET)
    result = svc.encode(data, expire_seconds)
    return result

def decodeToken(token: str|None, SECRET: str):
    """
    解密 token
    :param token: token
    :param SECRET: 密钥
    :return: 解密后的数据
    """
    svc = JwtService(SECRET)
    if token is  None:
        log.warn("token is None")
        return None
    result = svc.decode(token)
    if result is None or 'data' not in result:
        return None
    return result["data"] 


