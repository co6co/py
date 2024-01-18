import asyncio
from threading import Thread

class NewThread:
    _loop:asyncio.AbstractEventLoop=None
    def __init__(self) -> None:
        self._loop=asyncio.new_event_loop()
        pass


    def start_loop(self):
        asyncio.set_event_loop()
        self._loop.run_forever() 
        

tts_loop = asyncio.new_event_loop()
thread_tts = Thread(target=start_loop, args=(tts_loop,))
thread_tts.start() # 启动tts线程