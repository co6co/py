from model.services import RTSPService

import pytest,asyncio

# pip install pytest-asyncio
@pytest.fixture #提供测试所需的预设数据、环境或资源
def rtsp_url():
    return "rtsp://admin:123456@192.168.3.1/media/video1"
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
