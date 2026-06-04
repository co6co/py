
from functools import wraps 
from sanic.request import Request
 
from co6co.data.result import Result
from co6co_sanic_ext.view_model import response_json 
from ...services.authService import AuthService,PermissionValid
 
async def checkApi(request: Request):
    """
    查询当前用户的对当前API是否有权限
    """
    check = PermissionValid(request)
    await check.init()
    return check.check()


def authorized(f):
    """
    认证
    认证不通过 返回 403,不在执行f 函数
    """
    @wraps(f)
    async def decorated_function(request: Request, *args, **kwargs):
        valid=await AuthService(request).validToken()
        if not valid:
            return response_json(Result.fail(message="token invalid or expire"), status=401) 
        valid = await checkApi(request)
        if valid: 
            response = await f(request, *args, **kwargs)
            return response
        else:
            # the user is not authorized.
            return response_json(Result.fail(message="not_authorized"), status=401)
    return decorated_function


def ctx(f):
    """
    设置请求上下文
    有token 将可以再上下文中获取 用户Id等信息 
    authorized 少api检查权限
    """
    @wraps(f)
    async def decorated_function(request: Request, *args, **kwargs):
        valid=await AuthService(request).validToken()
        if not valid:
            return response_json(Result.fail(message="token invalid or expire"), status=401)
        response = await f(request, *args, **kwargs)
        return response

    return decorated_function

