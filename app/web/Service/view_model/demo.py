
from sanic.response import text
from sanic import Request
from co6co.data.result import Result  
from co6co_permissions.view_model.base_view import   BaseMethodView 


class testView(BaseMethodView):
    routePath = "/test"

    async def get(self, request: Request, *args, **kvgargs):
        return self.response_json(Result.success(data={"a": "12"}))
