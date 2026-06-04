
from sqlalchemy.sql import Select
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.tables import DynamicCodePO
from co6co_permissions.model.enum import dict_state


class componentViews(AuthMethodView):
    routePath = "/<code:str>"

    async def get(self ):
        """
        获取组件代码
        """
        code = self.match_info.get("code")
        select = Select(DynamicCodePO.sourceCode).filter(DynamicCodePO.code == code, DynamicCodePO.state == dict_state.enabled.val)
        return await self.get_one( select, isPO=False)
