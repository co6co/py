from co6co.data.result import Result
from co6co_db_ext.actuator import Actuator
from co6co_db_ext.db_session import db_service
from co6co_db_ext.jwt_service import JwtService
from co6co_aio.server import AppConfig, _appStart
import pytest
from aiohttp import web
from co6co.utils import log

# from aiohttp.web import Request # 不用直接使用该对象
from utils import make_json_request, to_result
import json
import asyncio
from co6co_web_session.session import Session
from co6co_aio.viewbase import ViewBase


class View(ViewBase):
    async def post(self):
        return self.response_json(Result.success("成功"))


async def params():
    d={}
    print(d.get("d"))
    app = web.Application()
    dd=app.get("ddd")
    print(isinstance( app,dict))
    print(dd)
    
    Session.mount_session(app)
    request = make_json_request(
        method="POST",
        path="/users/1",
        headers={"Content-Type": "application/json"},
        app=app,
        json_data={"nickName": "单元测试", "name": "单元测试",
                   "code": "fbb49e46-e598-4bb2-a48b-90940a16cabf", "passwd": "123456"},
    )
    return request


async def port_post():
    try:
        request, actuator = await params()
        view = View(request, actuator)
        response = await view.post()
        to_result(response)
    except Exception as e:
        log.err(e)
        raise e
    finally:
        pass


def test_post():
    asyncio.run(port_post())
