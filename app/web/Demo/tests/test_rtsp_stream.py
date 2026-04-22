from model.services import RTSPService

import pytest,asyncio
from model.apphelp import get_config

# pip install pytest-asyncio
@pytest.fixture #提供测试所需的预设数据、环境或资源
def rtsp_url():
    config=get_config()
    url=config.get("rtsp_url")
    return url
@pytest.fixture #提供测试所需的预设数据、环境或资源
def key():
    return "test_key"

@pytest.mark.asyncio
async def test_read_data(rtsp_url,key): 
    service=RTSPService()
    gen=  service.read_rtsp_stream(rtsp_url,key)
    assert gen!=None
    print(gen,"123456")
    async for data in gen:
        print(data) 
