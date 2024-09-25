from co6co_permissions.view_model.base_view import AuthMethodView
from sanic.response import text, raw
from typing import List, Optional
from sanic import Request
from view_model.wx import wx_authon_views
from model.wx import jsApiParam, jsApiResult
from services.wx_js_api_service import create_jsapi_signature, check_jsapi_signature
from co6co_sanic_ext.model.res.result import Result
from co6co.utils import log
from wechatpy.exceptions import WeChatClientException


class View(wx_authon_views):
    """
    jsAPI
    {"url":str}
    """

    async def post(self, request: Request):
        try:
            param = jsApiParam()
            param.__dict__.update(request.json)
            # log.warn(param.url)
            client = await self.getWxClient(request)
            if client == None:
                return self.response_json(Result.fail(message="未能查询到微信客户端！"))
            log.start_mark("获取jsapi ticket...")
            ticket = client.jsapi.get_jsapi_ticket()
            log.end_mark("获取jsapi ticketed:{}".format(ticket))
            result = jsApiResult()
            result.appId = client.appid
            result.signature = create_jsapi_signature(ticket, result.timestamp, result.nonceStr, param.url)
            # data = check_jsapi_signature(ticket, result.timestamp, result.nonceStr, param.url, result.signature)
            # log.warn("验证签名：", data, result.signature, result)
            return self.response_json(Result.success(data=result))
        except WeChatClientException as e:
            log.warn("jspai error:", e)

            return self.response_json(Result.fail(message="{}:{}".format(e.errcode, e.errmsg)))
        except Exception as e:
            log.err("jspai error2:", e)
            return self.response_json(Result.fail(message=e))
