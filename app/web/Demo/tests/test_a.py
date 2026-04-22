import pytest,asyncio

from a import get_data,get_json_data,get_rtsp_stream_data
from model.apphelp import get_config

@pytest.mark.asyncio
async def test_data_1():
    """需要等待所有生成器完成后才会结束 """
    async for data in get_data():
        print(data)

def test_json_data_1():
    """需要等待所有生成器完成后才会结束 """
    async def test():
        async for data in get_json_data():
            print(data)
    asyncio.run(test())

def test_rtsp_stream_data_1():
    """需要等待所有生成器完成后才会结束 """
    async def test():
        config=get_config()
        url=config.get("rtsp_url")
        async for data in get_rtsp_stream_data(url):
            print(data)
    asyncio.run(test())