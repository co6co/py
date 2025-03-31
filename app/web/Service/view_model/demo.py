
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.model.res.result import Result
from co6co_sanic_ext.utils import JSON_util
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView, BaseMethodView
from model.pos.tables import DynamicCodePO
from view_model._filters.sysTask import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute


class testView(BaseMethodView):
    routePath = "/test"

    async def get(self, request: Request, *args, **kvgargs):
        return self.response_json(Result.success(data={"a": "12"}))
