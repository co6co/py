import asyncio
from services.ff_service import ffService
ff_service = ffService()

async def main():
    async for data in ff_service.read_rtsp_stream('rtsp://admin:lanbo12345@192.168.3.1/media/video1','1'):
        print(data)


asyncio.run(main())
   
