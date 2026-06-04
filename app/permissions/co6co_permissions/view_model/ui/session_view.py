

from sanic import Request
from co6co.data.result import Result 
from co6co_web_db.view_model import BaseDbClsView 
import uuid


class Session_View(BaseDbClsView):
    routePath = "/"

    async def post(self, request: Request):
        """
        获取用户Session
        """
        session, _ = self.get_Session(request)
        # data = await jwt_service.createToken(getSecret(request), str(uuid.uuid4()),  session.expiry)
        return self.response_json(Result.success(data={"data":  str(uuid.uuid4()), "expiry": session.expiry}))
