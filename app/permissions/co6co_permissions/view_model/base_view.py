from co6co_web_db.view_model import BaseMethodView,Request

from .aop.api_auth import authorized
from co6co_db_ext import db_operations
from co6co.utils import log 

class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 
   
   def getUserId(self, request: Request):
      """
      获取用户ID
      """  
      return request.ctx.current_user["id"]  
      