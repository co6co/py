from aiohttp.test_utils import make_mocked_request
from aiohttp.streams import StreamReader
from aiohttp import web
import asyncio
import json
from unittest.mock import Mock
from co6co.data.result import Result
from co6co.utils import log

def to_result(resposne: web.Response):
    assert resposne.status == 200
    data = resposne.body
    JsonData = json.loads(data)
    log.warn(JsonData)
    res = Result.success()
    res.__dict__.update(JsonData)
    print(res.message)
    assert res.code == 0
    return res


def make_json_request(
    method="GET", path="/", json_data=None, headers=None, app=None, loop=None
):
    """兼容所有 aiohttp 版本的 JSON 请求构造"""
    if loop is None:
        loop = asyncio.get_running_loop()

    final_headers = headers or {}
    payload = None
    mock_protocol = Mock()
    if json_data is not None:
        payload = StreamReader(protocol=mock_protocol, limit=2**16, loop=loop)
        payload.feed_data(json.dumps(json_data).encode())
        payload.feed_eof()
        final_headers["Content-Type"] = "application/json"

    return make_mocked_request(
        method=method, path=path, headers=final_headers, payload=payload, app=app
    )
