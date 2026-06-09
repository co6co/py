from functools import wraps
from ...model.pos.right import LoginLogPO

from co6co_sanic_ext.view_model import response_json
from datetime import datetime
from sanic.response import JSONResponse

from co6co.utils import log
from co6co.data.result import Result
import json
import time
from ...configs.captcha import CaptchaConfig
from co6co_web_session.base import SessionDict
from ..base_view import AbsClsView
from ...services.utils import appHelper


async def _loginLog(response: JSONResponse, self: AbsClsView):
    try:
        po = LoginLogPO()
        po.ipAddress =self. request.client_ip  # p.ip=self.forwarded['for']
        po.createTime = datetime.now()
        res = json.loads(str(response.body, encoding="utf-8"))
        result = Result.success()
        result.__dict__.update(res)
        po.name =self. request.json.get("userName")
        if result.code == 0:
            po.createUser = appHelper.current_user_id(self.request)
            po.state = "成功"
        else:
            po.state = "失败"
        # log.warn(po.__dict__) 
        self.actuator.add_all(po)
        
    except Exception as e:
        log.err("写登录日志失败",e)


def loginLog(f):
    @wraps(f)
    async def decorated_function(self,*args, **kwargs):
        if not isinstance(self, AbsClsView):
            raise Exception("登录日志装饰器只能用于AbsClsView类")
        """
        for a,v in kwargs:
            log.warn("第er个参数",type(a),type(v))
        """
        response = await f(self,*args, **kwargs)
        await _loginLog(response,self)
        return response 
    return decorated_function


def _checkVerifycode(sessionDict:SessionDict,verifyCode: str=None,):
    """
    检查验证码
    """ 
    if verifyCode == "" or verifyCode is None:
        log.warn("验证码不能为空！")
        return False, "验证码不能为空！" 
    # 方案1 拖拉方式验证 
    memCode = sessionDict.get("verifyCode", None)
    if memCode:
        # 如果没有 key
        memCode = sessionDict.pop("verifyCode")
        if memCode != verifyCode:
            return False, "验证码错误！"
        return True, "验证成功！"
    # 方案2 验证码方法
    memCode = sessionDict.get("captcha_code", None)
    if not memCode:
        return False, "验证码不存在！,刷新页面重试！"
    stored_code: str = sessionDict.pop("captcha_code", "")  # 使用pop移除会话中的验证码
    # log.warn("验证码",stored_code)
    stored_timestamp = sessionDict.pop("captcha_timestamp", 0)

    # 检查验证码是否存在
    if not stored_code:
        return False, "验证码不存在！"
    # 检查验证码是否过期
    if int(time.time()) - stored_timestamp > CaptchaConfig.EXPIRE_SECONDS:
        # log.warn("验证码已过期，请重新获取！")
        return False, "验证码已过期，请重新获取！"
    # 检查验证码是否匹配
    if not CaptchaConfig.CASE_SENSITIVE:
        stored_code = stored_code.lower()
        verifyCode = verifyCode.lower()
    result = stored_code == verifyCode
    # log.warn("验证码匹配结果",stored_code == verifyCode,stored_code ,verifyCode)
    return result, "验证成功！" if result else "验证码错误！"


def verifyCode(f):
    """
    验证码装饰器
    """ 
    @wraps(f)
    async def _function(self,*args, **kwargs): 
        if isinstance(self, AbsClsView): 
            dict=self.json.get("verifyCode")
            _, sessionDict = self.get_Session(self.request)
            result, msg = _checkVerifycode(sessionDict,dict)
            if not result:
                log.warn(msg, result)
                return response_json(Result.fail(message=msg))
        else:
            raise Exception("验证码装饰器只能用于AbsClsView类")
        return await f(self,*args, **kwargs)
    return _function
