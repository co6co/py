import asyncio
from model.services import RTSPService
from co6co.utils import log
from co6co.task.thread import create_event_loop
import threading

async def test():
    service = RTSPService()
    test2 = service.read_rtsp_stream(
        "rtsp://admin:bwk68240175@anhctxz.com:65019/stream1", "123"
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

def test2():
    def demo(LOOP:asyncio.AbstractEventLoop  ): 
        LOOP.create_task(test( )) 
        LOOP.run_forever()  # 必须启动事件循环
        pass
    def createThread( ):
        LOOP=create_event_loop()
        thread=threading.Thread(target=demo,args=(LOOP ,) )
        thread.start()
        return LOOP
    log.start_mark("demo 线程")
    thread_task=createThread( )

    log.end_mark("demo 线程")
    
if __name__ == "__main__":
    #asyncio.run(test())
    test2()
    print("done")
