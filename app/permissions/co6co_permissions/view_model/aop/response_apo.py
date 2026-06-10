from functools import wraps
from co6co_web_db.view_model import BaseDbClsView
from sanic.response import  BaseHTTPResponse
from co6co .data.result import Result,Page_Result

def response(f):
    """
    web 响应aop 
    放在该模块 可以做其他事情
    """ 
    @wraps(f)
    async def _function(self, *args, **kwargs):
        if isinstance(self, BaseDbClsView):
            value = await f(self, *args, **kwargs) 
            if isinstance(value, BaseHTTPResponse):
                return value
            elif isinstance(value, Result) or isinstance(value, Page_Result):
                if value.code != 0:
                    self.set_rollback( ) 
            return self.response_json(value) 
        else:
            raise ValueError("unoverride response , BaseClsView must be defined") 
    return _function
