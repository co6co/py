import aiohttp.web as web
 
from co6co.utils.json_util import   JSONEncoder
from co6co.data.result import Result, Page_Result  
from co6co_db_ext.db_session import db_service
from co6co_db_ext.actuator import Actuator

class ViewBase(web.View):
    route = "/api/v0" 
    def __init__(self, request: web.Request, actuator:Actuator):
        self._session = actuator.session
        self._actuator = actuator
        self.db: db_service = request.app.db
        super().__init__(request)
        self._jwt:JwtService=request.app.jwtService 
    @property
    def jwtService(self):
        return self._jwt
    @property
    def Session(self):
        """
        会话工厂
        #async with db.Session() as session:
        #    ....
        #
        """
        return self.db.Session

    @property
    def session(self):
        return self._session
    @property
    def actuator(self):
        return self._actuator
    @classmethod
    def response_json(self, data: Result | Page_Result, status=200):
        return web.json_response(
            data,
            content_type="application/json",
            status=status,
            dumps=JSONEncoder.dumps,
        ) 

class ViewBaseAuth(ViewBase):
    
    """
    认证视图模型
    """
    class UserData:
        """
        认证数据
        """
        def __init__(self, userData:dict):
            self._userData = userData
            self.userId = userData["userId"]
            self.userName = userData["userName"]
            self.role = userData["role"] 
    route = "/api/v0/auth"
    def __init__(self, request, actuator:Actuator, userData:dict):
        self._userData = userData
        super().__init__(request, actuator)
    @property
    def userData(self):
        return self.UserData(self._userData)
