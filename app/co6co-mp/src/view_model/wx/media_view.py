from co6co_permissions.view_model.base_view import AuthMethodView
from sanic.response import text, raw
from typing import List, Optional
from sanic import Request


class Media_View(AuthMethodView):
    """
    素材管理
    """
    async def post(request: Request):
        request.raw_headers
        return text("")
