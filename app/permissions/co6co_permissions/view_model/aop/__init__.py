

from sanic.request import Request
from ...model.pos.right import UserPO
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result


def getCtxUserId(request: Request):
    if "current_user" in request.ctx.__dict__.keys():
        return request.ctx.current_user["id"]
    return None


def getCtxData(user: UserPO):
    """
    通过user获取 dict 保存在 request.ctx.current_user 中 
    """
    return user.to_jwt_dict()


ObjectExistRoute = "/exist/<code:str>/<pk:int>"


def exist(isExist: bool = True, tableName="用户", name: str = "xxx"):
    if isExist:
        return JSON_util.response(Result.success(data=True, message=f"{tableName}'{name}'已存在。"))
    else:
        return JSON_util.response(Result.success(data=False, message=f"{tableName}'{name}'不已存在。"))
