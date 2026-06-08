
from sanic.response import text
from sanic import Request
from co6co.data.result import Result

from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from services.cfService import CfService
from co6co.utils import log


class ListView(AuthMethodView):

    async def init(self):
        self.service = await CfService.instance(self.request)

    def isSucc(self, data: dict):
        log.warn(data)
        errors: list = data.get("errors", [])
        error = errors[0] if errors and len(errors) else None
        errMsg = f"{error.get("code", None)}- {error.get("message", "")}" if error else None
        return data.get("success", False), errMsg

    def res(self, data: dict):
        succ, msg = self.isSucc(data)
        return self.response_json(Result.success(data=data)) if succ else self.response_json(Result.fail(message=msg))

    async def get(self ):
        """
        获取dns记录
        """
        try:
            await self.init( )
            data = self.service.list_dns_records()
            # log.warn(data)
            return self.res(data)
        except Exception as e:
            log.err(e)
            return self.response_json(Result.fail(message=str(e)))

    async def put(self ):
        """
        创建dns记录
        param:{ 
            "type":A: ipv4|AAAA: ipv6|CNAME: 域名 
            "name": "www", # 子域名
            "content": "1.1.1.1",
            "ttl": 1,
            "proxied": true,
            "comment": str = None,
        } 
        """
        await self.init( )
        keys = ['type', 'name', 'content', 'ttl', 'proxied', "comment"]
        parme = self.choose(self. request, keys, True)
        data = self.service.create_dns_record(**parme)
        return self.res(data)

    async def patch(self ):
        """
        更新dns记录
        {
            record_id: str,
            type: str,
            name: str,
            content: str,
            ttl: int = 1,
            proxied: bool = False,
            comment: str = None,
        }
        """
        await self.init( )
        keys = ["record_id", 'type', 'name', 'content', 'ttl', 'proxied', "comment"]
        parme = self.choose( keys, True)
        data = self.service.update_dns_record(**parme)
        return self.res(data)


class OneView(ListView):
    routePath = "/<id:str>"
    @property
    def guid(self):
        return self.match_info["id"]
                
    async def get(self ):
        """
        获取dns记录
        """
        await super().init()
        data = self.service.get_dns_record(self.    guid)

        return self.res(data)

    async def delete(self ):
        """
        删除dns记录"
        """
        await super().init( )
        data = self.service.delete_dns_record(self.guid)
        return self.res(data)
