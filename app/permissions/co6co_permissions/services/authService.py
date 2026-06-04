
from co6co_db_ext.appconfig import AppConfig 
from sanic.request import Request
 
from co6co.utils import log 
from .baseCache import CustomSanicCache
from co6co_db_ext.jwt_service import JwtService
from ..model.pos.right import UserPO
from ..model.enum import user_state
from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import  db_tools
from typing import List, Dict  
from .authCache import AuthonCacheManage

class AuthService:
    """
    认证服务
    """
    def __init__(self,request: Request) -> None:
        self.request = request
        self.appconfig=AppConfig.get_config( request.app.config)
        jwt_secret = self.appconfig.raw.get("SECRET")
        self._jwtService = JwtService(jwt_secret)
        self._sanicCache = CustomSanicCache(self.request)
    @property
    def jwtService(self) -> JwtService:
        return self._jwtService
    @property
    def sanicCache(self) -> CustomSanicCache:
        return self._sanicCache
    @property
    def token(self):
        return self.request.token
    
    def setAuthonContext(self, data):
        """
        设置用户上下文
        :param user: 用户信息
        :type user: UserPO
        :return: None
        :rtype: None
        """
        self.request.ctx.current_user = data
    async def _validAccessToken(self) : 
        result = self.sanicCache.getCache(self.token)
        if result:
            self.setAuthonContext( result)
            return True
        
        select = Select(UserPO).filter(UserPO.password.__eq__(self.token), UserPO.state.in_([user_state.enabled]))
        user: UserPO = await db_tools.execForPo(self.sanicCache.dbSession, select, remove_db_instance_state=False)
        
        if user is None:
            log.warn("query {} accessToken is NULL".format(self.token))
            return False
        else:
            data = user.jwt_data
            self.sanicCache.setCache(self.token, data)
            self.setAuthonContext(  data)
            return True
    def _validjwtToken(self):
        data= self.jwtService.decode(  self.token )
        if data is None:
            log.warn("decode {} jwt token failed".format(self.token))
            return False
        self.setAuthonContext(  data)
        return True
    async def validToken(self): 
        token=self.token
        if token and '.' not in token:
            return await self._validAccessToken()
        elif token: 
            return self._validjwtToken()
        return False



class PermissionValid:
    request: Request = None
    currentUserMenus: List[Dict] = None
    inited: bool = False

    def __init__(self, request: Request) -> None:
        self.request = request
        pass
    # 协调函数

    async def init(self):
        """
        初始化
        """
        cacheManage = AuthonCacheManage(self.request)
        allMenuData = await cacheManage.menuData
        currentUserRoles = await cacheManage.currentRoles
        self.currentUserMenus = []
        [self.currentUserMenus.append(m) for m in allMenuData if m.get("roleId") in currentUserRoles and m.get("id") not in map(lambda m: m['id'], self.currentUserMenus)]
        self.inited = True
    # def __await__(self):
        # 需要生成器对对象
        # allMenuData= yield from  cacheManage.menuData

    def check(self) -> bool:
        if not self.inited:
            log.err("未初始化.")
            return False
        for menu in self.currentUserMenus:
            if self._check(menu):
                return True
        return False

    def _check(self, menu: Dict):
        url: str = menu["url"]
        path = self.request.path
        method = self.request.method
        methods: str = menu["methods"]
        methods: list = methods.split(",")
        if method not in methods and "ALL" not in methods:
            return False
        pathArr = path.split("/")
        if "**" in url:
            url = url[0:url.index("**")]
            urlArr = url.split("/")
            # log.warn(pathArr,urlArr)
            if len(pathArr) >= len(urlArr)-1 and pathArr[0:len(urlArr)-1] == urlArr[0: len(urlArr)-1]:
                return True
        if "*" in url:
            url = url[0:url.index("*")]
            urlArr = url.split("/")
            #log.warn(pathArr, urlArr)
            if (len(pathArr) == len(urlArr) or len(pathArr) == len(urlArr)-1) and pathArr[0:len(urlArr)-1] == urlArr[0: len(urlArr)-1]:
                return True
        if url == path:
            return True
        return False
