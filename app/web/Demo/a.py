import asyncio
from model.services import RTSPService

if __name__ == "__main__":
   
    async def test(): 
        service=RTSPService()
        #test2= service.read_rtsp_stream("rtsp://admin:lanbo12345@192.168.3.1/media/video1",'123')
        test2= service.exec_ipconfig()
        async for data in test2:
            print(data.decode('gbk'))
    asyncio.run(test())