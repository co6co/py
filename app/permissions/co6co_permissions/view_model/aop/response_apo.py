from functools import wraps
from co6co_sanic_ext.view_model import BaseClsView
from sanic.response import  BaseHTTPResponse

def response(f):
    """
    web 响应aop 
    放在该模块 可以做其他事情
    """ 
    @wraps(f)
    async def _function(self, *args, **kwargs):
        if isinstance(self, BaseClsView):
            value = await f(self, *args, **kwargs)
            if isinstance(value, BaseHTTPResponse):
                return value
            else:
                return self.response_json(value)
        else:
            raise ValueError("unoverride response , BaseClsView must be defined") 
    return _function
