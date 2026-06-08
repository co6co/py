
from sanic import Request
from co6co.data.result import Result
import re
from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import QueryOneCallable
from co6co_permissions.view_model.base_view import AuthMethodView

from view_model.tools import data
from co6co_permissions.model.pos.other import sysDictPO


class View(AuthMethodView):
    # 类别为字典Id
    routePath = "/<category:int>"
    def category(self):
        return self.match_info.get("category") 
    async def get(self ):
        category = self.category()
        from co6co.log import log
        log.warn( "category 测试<category:int> ", type( category))
        select = (
            Select(sysDictPO.flag, sysDictPO.id)
            .filter(sysDictPO.id.__eq__(category))
        )
        dictOne: dict = await self.actuator.query_one_mappings(select)
         
        value = dictOne.get("flag", "")
        return self.response_json(Result.success(data.toDesc(value)))

    async def post(self ):
        try:
            json: dict = self.json
            lst = json.get("list")
            danList: list = json.get("dans", [])
            select = (
                Select(sysDictPO.desc)
                .filter(sysDictPO.id.__eq__(self.category))
            )
            dictOne: dict = await self.actuator.query_one_mappings(select) 
            desc: str = dictOne.get("desc", "")
            arr = re.split(r'\r\n|\r|\n', desc)
            arr = [a.replace(' ', '').split(',') for a in arr]
            rest = data.Padding(lst, arr, *danList)
            return self.response_json(Result.success({"list": rest, "count": len(rest)}))
        except Exception as e:
            return self.response_json(Result.fail(message=f"异常：{e}"))
