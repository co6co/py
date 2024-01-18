from view_model.base_view import BaseMethodView,Request 
from sanic.response import text,raw
from co6co .utils import log
class TestView(BaseMethodView):
    def get(self,request:Request ):
        data=request.app.ctx.data 
        request.app.ctx.data=request.app.ctx.data+1
        log.warn(f"*****Data:id:{id(data)},value:{data}") 
        return text(f"请求成功，你可以试试其他的:{request.args},{data}") 
class TestsView(BaseMethodView):
    def get(self,request:Request ):
        data=request.app.ctx.data
        log.warn(f"*****Data:id:{id(data)},value:{data}")  
        return text(f"请求成功，你可以试试其他的:{request.args},{data}") 
        