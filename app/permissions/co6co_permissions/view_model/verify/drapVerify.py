
from co6co_web_db.view_model import BaseDbClsView
from sanic import Request
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result
import uuid
import datetime


class drap_verify_view(BaseDbClsView):

    async def post(self):
        """
        拖动验证
        """
        json: dict = self.json
        start = json.get("start", 0)
        end = json.get("end", 0)
        data = json.get("data", [])
        start = datetime.datetime.fromtimestamp(start/1000)
        end = datetime.datetime.fromtimestamp(end/1000)
        dif = end-start
        min = datetime.timedelta(milliseconds=60)
        max = datetime.timedelta(seconds=15)
        if dif > min and dif < max:
            s = str(uuid.uuid4())
            _, sDict = self.get_Session(self.request)
            sDict["verifyCode"] = s
            return response_json(Result.success(data=s, message=f"验证成功,用时：{dif.total_seconds()}s"))
        else:
            return response_json(Result.fail(message="验证失败"))
