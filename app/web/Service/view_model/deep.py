
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
from co6co.enums import Base_Enum
from co6co_sanic_ext.utils import JSON_util
import threading
import time
import asyncio
import copy
from co6co_permissions.services.bllConfig import config_bll
from co6co.task.thread import ThreadEvent


class _dbView(BaseMethodView):
    async def query_config_value(self, request: Request, key: str, parseDict: bool = False) -> str | dict:
        config = config_bll()
        return config.run(config.query_config_value, key, parseDict)


class DeepseekView(_dbView):
    routePath = "/deep"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessionKey = "CHAT_DATA"

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

    def querySession(self, request: Request) -> Result | None:
        _, sessionDict = self.get_Session(request)
        if self.sessionKey in sessionDict:
            result: str = sessionDict.pop(self.sessionKey)
            s = Result.success()
            data = sysJson.loads(result)
            s.__dict__.update(data)
            return s

        return None

    def threadQuery(self, request: Request, execution_time: float = 5):

        # 启动一个线程来执行任务
        sess, sessionDict = self.get_Session(request)

        def run_async_loop():
            t = ThreadEvent("bkc_query")
            try:
                data = t.runTask(self.query, request)
                sessionDict = t.runTask(sess.interface.open, request)
                sessionDict.setdefault(self.sessionKey,  JSON_util(ensure_ascii=False).encode(data))
                t.runTask(sess.interface.save, request, {})
            finally:
                # loop.close()
                # t.close()
                log.err("线程退出。")
                pass
        thread = threading.Thread(target=run_async_loop)
        thread.start()
        # 等待一段时间，检查任务是否完成
        start_time = time.time()

        while time.time() - start_time < execution_time:
            if self.sessionKey in sessionDict:
                result: Result = self.querySession(request)
                # log.err("主函数推出.2")
                return result
            time.sleep(0.1)
        # log.err("主函数推出.")
        return None

    async def query(self, request: Request):
        """
        对  话
        """
        try:
            user_message = request.json.get("message")
            if not user_message:
                return Result.fail("Missing 'message' in request body")
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
            response = requests.post(self.api_url, headers=headers, json=data, proxies=self.proxys)
            if response.status_code == 200:
                result = response.json()
                request = copy.copy(request)
                message = result["choices"][0]["message"]["content"]
                Data = Result.success(data=message)
                return Data
            else:
                return Result(data=f"API request failed: {response.text}", code=response.status_code)
        except Exception as e:
            log.err("调用deepseekError", e)
            return Result(data=str(e), code=500)

    async def get(self, request: Request):
        data = self.querySession(request)
        return self.response_json(data if data else Result.success(message="没有数据，稍后重试！"))

    async def post(self, request: Request):
        """
        对  话
        """
        # return self.response_json(await self.query(request))
        log.err("int Id", id(request))
        data = self.threadQuery(request, 8)
        if not data:
            data = Result.success("数据复制在Session中.")
        return self.response_json(data)


class SgView(_dbView):
    routePath = "/sg"

    async def get(self, request: Request):
        """
        对  话
        """
        args = self.usable_args(request)
        deviceName = args.get("deviceName")
        return self.response_json(await self.queryStatue(request, deviceName))

    async def queryStatue(self, request: Request, deviceName: str):
        url = "https://songguoyun.topwd.top/Esp_Api_advance.php"
        result: dict = await self.query_config_value(request, "SONGGUO_API_USER", parseDict=True)
        result.update({"type": 1})
        response = requests.post(url,  json=result)
        if response.status_code == 200:
            result = unquote(response.text)
            result = sysJson.loads(result)
            if deviceName:
                result = result.get("deviceslist", [])
                result = [i for i in result if i.get("deviceName") == deviceName]
                result = result[0] if result else {}
                state = result.get("status")
                if state == 1:
                    result.update({"statusDesc": "设备已开机"})
                elif state == 0:
                    result.update({"statusDesc": "设备已关机"})
                elif state == 2:
                    result.update({"statusDesc": "设备离线"})
                elif state == 3:
                    result.update({"statusDesc": "设备在线"})

            result.update({"statusDescs": "0关机 1开机 2离线 3在线"})
            return Result.success(result, message=result.get("statusDesc"))
        else:
            return Result({"error": f"API request failed: {response.text}"}, code=response.status_code)

    async def exec(self, request: Request):
        """
        执行命令
        request.json->{deviceName:str,value:int}

        deviceName: 设备名称
        value:  0 关机                      0 成功，-1 失败
                1 开机                      0 成功，-1 失败
                2 强制重启                  0 成功，-1 失败
                11 查询状态                 1 电脑开机，0 电脑关机
                14 强制关机                 0 成功，-1 失败
                25 重启                     0 成功，-1 失败


        data:{'status': int, 'tips': str}
        """
        deviceName = request.json.get("deviceName")
        value: int = request.json.get("value")
        url = "https://songguoyun.topwd.top/Esp_Api_new.php"
        result: dict = await self.query_config_value(request, "SONGGUO_API_USER", parseDict=True)
        result.update({"value": value, "device_name": deviceName})
        response = requests.post(url,  json=result)
        if response.status_code == 200:
            result = unquote(response.text)
            result = sysJson.loads(result)
            return Result.success(result, message=result.get("tips") or "已经打开了" if value == 1 else "已经关闭了")
        else:
            return Result({"status": -999, "tips": f"API request failed: {response.text}"}, code=response.status_code)

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
        return self.response_json(await self.exec(request))


class TM_RESPONSE_CODE(Base_Enum):
    """
    天猫结果类型
    """
    SUCCESS = "SUCCESS", 0  # ：代表执行成功
    PARAMS_ERROR = "PARAMS_ERROR", 1  # ：代表接收到的请求参数出错
    EXECUTE_ERROR = "EXECUTE_ERROR", 2  # ：代表自身代码有异常
    REPLY_ERROR = "REPLY_ERROR", 3  # ：代表回复结果生成出错


class TmView(_dbView):
    routePath = "/tm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepseek = DeepseekView()
        self.sg = SgView()

    def getStandardValue(self, request: Request):
        """
        获取标准值

        json:{   'sessionId': '9802c885-3a63-4f44-96e0-54c03bd68e00', 
            'utterance': '关闭电脑', 
            'requestData': {}, 'botId': 158095, 'domainId': 87154, 'skillId': 110255, 'skillName': '智能 助手', 
            'intentId': 186451, 'intentName': 'custom_close', 
            'slotEntities': [
                {'intentParameterId': 138981, 'intentParameterName': 'deviceName', 'originalValue': '电脑', 'standardValue': 'home_PC', 'liveTime': 0, 'createTimeStamp': 1741833828738, 'slotName': 'deviceName:switch', 'slotValue': 'home_PC'}], 
            'requestId': '20250313104348694-1440851220', 'device': {}, 'skillSession': {'skillSessionId': '1b67b7ee-85e8-43e9-94c2-29f4456c84f6', 'newSession': True}, 'context': {'system': {'apiAccessToken': ''}}}
        """
        entities = request.json.get("slotEntities")
        value = entities[0].get("standardValue")
        return value

    async def exec(self, request: Request, value: int):
        """
         value: 0 关机 
                1 开机 
                2 强制重启 
                11 查询状态 1 电脑开机，0 电脑关机 
                14 强制关机 
                25 重启   
        """
        deviceName = self.getStandardValue(request)
        request.json.update({"deviceName": deviceName, "value": value})
        result = await self.sg.exec(request)
        log.warn("执行结果：", result)
        if result.code == 0:
            return await self.responseTm(result.message, TM_RESPONSE_CODE.SUCCESS)
        else:
            return await self.responseTm(result.message, TM_RESPONSE_CODE.REPLY_ERROR)

    async def responseTm(self, text, status: TM_RESPONSE_CODE):
        # 普通文本
        data = {
            "returnCode": str(status.val),
            "returnErrorSolution": "",
            "returnMessage": "",
            "returnValue": {
                "reply": "回复用户的普通文本内容",  # 回复给用户的普通文本信息
                "replyType": "TEXT",  # reply的类型，默认为 TEXT
                "resultType": "RESULT",  # 回复时的状态标识，当值为 RESULT 时，天猫精灵播放完回复内容后不会开麦
                # SUCCESS：代表执行成功
                # PARAMS_ERROR：代表接收到的请求参数出错
                # EXECUTE_ERROR：代表自身代码有异常
                # REPLY_ERROR：代表回复结果生成出错
                "executeCode": status.key
            }
        }
        # action 回复普通文本
        data = {
            "returnCode": str(status.val),
            "returnErrorSolution": "",
            "returnMessage": "",
            "returnValue": {
                "actions": [
                    {
                        "name": "playTts",  # Action名称，播放TTS文本内容时该名字必须设置为 playTts
                        # key: "content"，value为需要播报的内容；
                        # key: "format"，value为 text；
                        # key: "showText"，value为天猫精灵APP内设备对话记录展示的内容，一般与"content"一致即可
                        "properties": {
                            "content": "回复用户的普通文本内容",
                            "format": "text",
                            "showText": "回复用户的普通文本内容"
                        }
                    }
                ],
                "resultType": "RESULT",  # 回复时的状态标识，当值为 RESULT 时，天猫精灵播放完回复内容后不会开麦
                "executeCode": status.key  # 同普通文本
            }
        }
        # Speak 指令回复普通文本
        data = {
            "returnCode": str(status.val),
            "returnErrorSolution": "",
            "returnMessage": "",
            "returnValue": {
                "resultType": "RESULT",
                "gwCommands": [
                    {
                        "commandDomain": "AliGenie.Speaker",
                        "commandName": "Speak",
                        "payload": {
                            "type": "text",             # 回复的类型，默认 text
                            "text": text,        # 期待用户继续对话
                            "expectSpeech": True,       # 是否开麦, 默认 false
                            "needLight": True,          # 开麦时是否需要灯光提示用户
                            "needVoice": True,          # 开麦时是否需要声音提示用户
                            "wakeupType": "continuity"  # 如果开麦设置为continuity，不开麦则不要设置
                        }
                    }
                ],
                "executeCode": status.key
            }
        }
        return self.response_json(data)

    async def post(self, request: Request):
        """
        天猫精灵接口
        执行时间不能超过 2s 包括2s,否则超时
        """
        '''
        {"sessionId":"2a4e3f67-9ebc-4970-b17e-80fd3b40f006","utterance":"打开筱筱","requestData":{},"botId":158095,"domainId":87154,"skillId":110255,"skillName":"智能助手","intentId":186450,"intentName":"chat","slotEntities":[],"requestId":"20250311143452004-1277958919","device":{},"skillSession":{"skillSessionId":"9dc47fb2-14c4-42c8-ad56-c2e0327776da","newSession":true},"context":{"system":{"apiAccessToken":""}}}
        '''
        log.warn("天猫接口：参数", request.json)
        log.warn("天猫接口header:参数", request.headers)
        _, sDist = self.get_Session(request)

        sid = request.json.get("sessionId")
        log.warn("sid", sid)
        result: str = await self.query_config_value(request, "TM_ENTRY_KEY")
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header and auth_header.startswith('Bearer '):
            token: str = auth_header.split(' ', 1)[1]
        if result.strip() != token.strip():
            return await self.responseTm("未授权", TM_RESPONSE_CODE.PARAMS_ERROR)
        name = request.json.get("intentName")
        try:
            from co6co.task.utils import Timer
            value = self.getStandardValue(request)
            tim = Timer(value)
            tim.start()
            if name == "chat":
                value = self.getStandardValue(request)
                request.json.update({"message": value})
                if value == "答案" or value == "上一个问题的答案":
                    res = self.deepseek.querySession(request)
                    res = Result.success("还没有结果，请稍后超时，长时间没有结果请重新提问") if not res else res
                else:
                    # result = await self.deepseek.query(request)
                    res = self.deepseek.threadQuery(request, 1.5)  # 不能超过2s
                    if not res:
                        res = Result.success("执行时间过长，请稍后来查找答案")
                log.warn(res)
                if res.code == 0:
                    return await self.responseTm(res.data, TM_RESPONSE_CODE.SUCCESS)
                else:
                    return await self.responseTm(res.data, TM_RESPONSE_CODE.REPLY_ERROR)

            elif name == "custom_close":
                return await self.exec(request, 0)
            elif name == "custom_open":
                return await self.exec(request, 1)
            elif name == "custom_query":
                deviceName = self.getStandardValue(request)
                result = await self.sg.queryStatue(request, deviceName)
                return await self.responseTm(result.message, TM_RESPONSE_CODE.SUCCESS)
            return await self.responseTm("未知指令", TM_RESPONSE_CODE.PARAMS_ERROR)
        finally:
            tim.stop()
