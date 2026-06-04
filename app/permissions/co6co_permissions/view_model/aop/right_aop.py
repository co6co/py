

from functools import wraps  
from ...services.authCache import AuthonCacheManage  
from ..base_view import AuthMethodView
def menuChanged(f):
    """
    设置请求上下文
    有token 将可以再上下文中获取 用户Id等信息 
    """
    @wraps(f)
    async def decorated_function(self,*args, **kwargs): 
        if isinstance(self, AuthMethodView):
            cacheManage = AuthonCacheManage(self.request)
            cacheManage.setMenuDataInvalid() 
        return await f(*args, **kwargs)

    return decorated_function


def userRoleChanged(f):
    """
    设置请求上下文
    有token 将可以再上下文中获取 用户Id等信息 
    """
    @wraps(f)
    async def decorated_function(self,*args, **kwargs):
        if isinstance(self, AuthMethodView):
            cacheManage = AuthonCacheManage(self.request)
            cacheManage.setRolesInvalid()
            cacheManage.setMenuDataInvalid() 
        response = await f(*args, **kwargs)
        return response

    return decorated_function
