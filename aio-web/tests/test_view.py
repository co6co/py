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
import json, asyncio


async def params():
    config = AppConfig.get_config("dist/testconfig.json")
    db_factory = db_service(config.db)
    actuator = Actuator(db_factory.Session())
    app = web.Application()
    app.config = config
    app.db = db_factory  # _appStart()
    app.jwtService = JwtService(config.web.jwt_secret)

    request = make_json_request(
        method="POST",
        path="/users/1",
        headers={"Content-Type": "application/json"},
        app=app,
        json_data={"nickName": "单元测试","name":"单元测试","code":"fbb49e46-e598-4bb2-a48b-90940a16cabf","passwd":"123456"},
    )
    return request, actuator


async def port_post( ):
    try:
        request, actuator = await params()
        view = WxBindUserView(request, actuator)
        response = await view.post()
        to_result(response)  
    except Exception as e:
        log.err(e)
        raise e
    finally:
        await actuator.session.rollback() 
        pass
        # 不提交数据库是没有的
def test_post(): 
    asyncio.run(port_post( ))
