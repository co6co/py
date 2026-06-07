from sanic import Request
from co6co_web_db.services import get_db_service, get_cache, get_db_session


class appHelper:
    @staticmethod
    def current_user( request: Request):
        """
        获取当前用户信息
        :return: 当前用户信息
        :rtype: {"id": int, "user_name": str, "group_id": int}
        """
        if "current_user" in request.ctx.__dict__.keys():
            return request.ctx.current_user
        else:
            raise Exception("当前用户信息不存在")
    @staticmethod
    def set_current_user( request: Request,data:dict):
        """
        获取当前用户信息
        :return: 当前用户信息
        :data: {"id": int, "user_name": str, "group_id": int}
        
        """ 
        request.ctx.current_user = data

    @staticmethod
    def current_user_id( request: Request):
        user = appHelper.current_user(request)
        if user:
            userId = int(user["id"])
            return userId

    @staticmethod
    def current_user_name( request: Request):
        user = appHelper.current_user(request)
        if user:
            userName = user["user_name"]
            return userName

    @staticmethod
    def current_user_group_id( request: Request):
        user = appHelper.current_user(request)
        if user:
            userName = user["group_id"]
            return userName
    @staticmethod
    def get_app_param( request: Request): 
        return get_cache(request.app), get_db_session(request),get_db_service(request.app)

       
      
