import asyncio
from threading import Thread
from time import sleep, ctime
from co6co.utils import log

class ThreadEventLoop:
	"""
	Run Event Loop in different thread.
	"""
	@property
	def loop(self):
		return self._loop

	def __init__(self):
		self._loop =asyncio. new_event_loop()
		log.warn(f"ThreadEventLoop:{id(self._loop)}")
		Thread(target=self._start_background, daemon=True).start()

	def _start_background(self):
		asyncio.set_event_loop(self.loop)
		self._loop.run_forever()
		
	def runTask(self, tastFun , *args, **kwargs):
		log.warn(f"ThreadEventLoop22:{id(self._loop)}")
		task=asyncio.run_coroutine_threadsafe(tastFun(*args, **kwargs), loop=self._loop)
		return task.result()
		