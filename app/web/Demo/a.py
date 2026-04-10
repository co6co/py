import asyncio
from model.services import RTSPService


async def test():
    service = RTSPService()
    test2 = service.read_rtsp_stream(
        "rtsp://admin:lanbo12345@192.168.3.1/media/video1", "123"
    )

    # test2= service.exec_ipconfig()
    ff = 0
    async for data in test2:
        # print(data.decode('gbk'))
        if ff > 10:
            break
        print(len(data))
        ff += 1
    print("anext:",len( await anext(test2)))
    
    #await test2.aclose()

if __name__ == "__main__":
    asyncio.run(test())
    print("done")
