
from sanic.response import text, json
from sanic import Request
from co6co_sanic_ext.model.res.result import Result
from co6co_permissions.view_model.base_view import BaseMethodView


import requests
from sqlalchemy.sql import Select, Delete
from co6co_permissions.model.pos.other import sysConfigPO
from co6co.utils import log
from co6co_db_ext.db_utils import QueryOneCallable
from urllib.parse import unquote
import json as sysJson


class _dbView(BaseMethodView):
    async def query_config_value(self, request: Request, key: str, parseDict: bool = False) -> str | dict:
        select = (
            Select(sysConfigPO.value)
            .filter(sysConfigPO.code.__eq__(key))
        )

        db = self.get_db_session(request)
        call = QueryOneCallable(db)
        result = await call(select, isPO=False)
        result: str = result.get("value")
        if parseDict:
            result = sysJson.loads(result)
        return result


class DeepseekView(_dbView):
    routePath = "/deep"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def init(self, request: Request, message: str):

        result: str = await self.query_config_value(request, "DEEPSEEK_KEY")
        result = result.replace("$$MESSAGE", message.replace("\"", "\\\""))
        result = sysJson.loads(result)
        self.api_key = result.get("api_key", None)
        self.api_url = result.get("api_url", None)
        if not self.api_key or not self.api_url:
            raise Exception("未配置API: api_key 或者 api_url")
        self.ext_header = result.get("ext_header", None)
        self.data = result.get("data", None)
        self.proxys = result.get("proxys", None)

    async def post(self, request: Request):
        """
        对  话
        """
        try:
            user_message = request.json.get("message")
            if not user_message:
                return json({"error": "Missing 'message' in request body"}, status=400)
            await self.init(request, user_message)
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                # "cf-aig-authorization": f"Bearer 2d4feD_Uc1zstZ2xI01M1_QUghBGvhS_JX2OKXw9",
                "Authorization": f"Bearer {self.api_key}"
            }
            headers.update(self.ext_header)
            # data demo
            '''
            data = {
                "model": "deepseek-r1-distill-qwen-32b",
                "store": True,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            }'
            '''
            data = self.data
            # 发送请求到 DeepSeek API
            response = requests.post(self.api_url, headers=headers, json=data, proxies=self.proxys)
            if response.status_code == 200:
                result = response.json()
                return json({"reply": result["choices"][0]["message"]["content"]})
            else:
                return json({"error": f"API request failed: {response.text}"}, status=response.status_code)
        except Exception as e:
            log.err("error", e)
            return json({"error": str(e)}, status=500)


class SgView(_dbView):
    routePath = "/sg"

    async def get(self, request: Request):
        """
        对  话
        """
        url = "https://songguoyun.topwd.top/Esp_Api_advance.php"
        result: dict = await self.query_config_value(request, "SONGGUO_API_USER", parseDict=True)
        result.update({"type": 1})

        response = requests.post(url,  json=result)
        if response.status_code == 200:
            result = unquote(response.text)
            result = sysJson.loads(result)
            result.update({"statusDesc": "0关机 1开机 2离线 3在线"})
            return json(result)
        else:
            return json({"error": f"API request failed: {response.text}"}, status=response.status_code)

    async def post(self, request: Request):
        """
        deviceName: 设备名称
        value:  0 关机                      0 成功，-1 失败
                1 开机                      0 成功，-1 失败
                2 强制重启                  0 成功，-1 失败
                11 查询状态                 1 电脑开机，0 电脑关机
                14 强制关机                 0 成功，-1 失败
                25 重启                     0 成功，-1 失败
        """
        deviceName = request.json.get("deviceName")
        value = request.json.get("value")
        url = "https://songguoyun.topwd.top/Esp_Api_new.php"
        result: dict = await self.query_config_value(request, "SONGGUO_API_USER", parseDict=True)
        result.update({"value": value, "device_name": deviceName})
        response = requests.post(url,  json=result)
        if response.status_code == 200:
            result = unquote(response.text)
            result = sysJson.loads(result)
            return json(result)
        else:
            return json({"error": f"API request failed: {response.text}"}, status=response.status_code)


class TmView(BaseMethodView):
    routePath = "/tm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepseek = DeepseekView()
        self.sg = SgView()

    async def post(self, request: Request):
        """
        天猫精灵接口
        """
        '''
        {"sessionId":"2a4e3f67-9ebc-4970-b17e-80fd3b40f006","utterance":"打开筱筱","requestData":{},"botId":158095,"domainId":87154,"skillId":110255,"skillName":"智能助手","intentId":186450,"intentName":"chat","slotEntities":[],"requestId":"20250311143452004-1277958919","device":{},"skillSession":{"skillSessionId":"9dc47fb2-14c4-42c8-ad56-c2e0327776da","newSession":true},"context":{"system":{"apiAccessToken":""}}}
        '''
        name = request.json.get("intentName")
        device = request.json.get("device")
        if name == "chat":
            request.json.update({"message": request.json.get("utterance")})
            return await self.deepseek.post(request)
        elif name == "custom_close":
            request.json.update({"deviceName": device.name, "value": 0})
            return await self.sg.post(request)
        elif name == "custom_close":
            request.json.update({"deviceName": device.name, "value": 1})
            return await self.sg.post(request)
        return json({"error": f"未能处理，该意图{name}"}, status=500)
