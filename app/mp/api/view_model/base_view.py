from co6co_sanic_ext.view_model import BaseMethodView 
from services import authorized

class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 