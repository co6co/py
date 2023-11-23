from co6co_sanic_ext.view_model import BaseMethodView,Request 
from sanic.response import text,raw
from co6co .utils import log
class TestView(BaseMethodView):
    def get(self,request:Request ):
        return text(f"请求成功，你可以试试其他的:{request.args}")