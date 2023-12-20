from co6co_web_db.view_model import BaseMethodView, Request
from services import authorized


class AuthMethodView(BaseMethodView):
    decorators = [authorized]

    def getUserId(self, request: Request):
        """
        获取用户ID
        """
        return request.ctx.current_user["id"]
