import pytest,asyncio

from a import get_data,get_json_data,get_rtsp_stream_data

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
        async for data in get_rtsp_stream_data("rtsp://admin:admin@192.168.1.64:554/Streaming/Channels/101"):
            print(data)
    asyncio.run(test())