import asyncio
from threading import Thread
from time import sleep, ctime
from co6co.utils import log

class ThreadEvent:
	"""
	线程Event loop 
	Run Event Loop in different thread.
	
	## 因某些原因写了该类 
	## 1. asyncio.run 在 没有正在运行的事件循环 的情况下运行协程的
    ##wx_user_dict:list[dict]=asyncio.run(bll.get_subscribe_alarm_user(config.appid)) 
    ##alarm= alarm_bll(app)
    ##po:bizAlarmTypePO=asyncio.run(alarm.get_alram_type_desc(po.alarmType))
    ## 2. 正在运行的事件循环
    ## This event loop is already running   
    ## import nest_asyncio 
    ##loop = asyncio.get_event_loop()
    ##wx_user_dict:list[dict]=loop.run_until_complete(bll.get_subscribe_alarm_user(config.appid)) 
	
    ## 3. 创建任务
    ##task=asyncio.create_task(bll.get_subscribe_alarm_user(config.appid))
    ##wx_user_dict:list[dict]=asyncio.run(task) 
    ## 4.底层使用
    ##asyncio.ensure_future(coro())
    ## 5. 
	"""
	@property
	def loop(self):
		return self._loop

	def __init__(self):
		self._loop =asyncio.new_event_loop() 
		#log.warn(f"ThreadEventLoop:{id(self._loop)}")
		Thread(target=self._start_background, daemon=True) .start()
		

	def _start_background(self):
		asyncio.set_event_loop(self.loop)
		self._loop.run_forever()
		
	def runTask(self, tastFun , *args, **kwargs):
		#log.warn(f"ThreadEventLoop22:{id(self._loop)}")
		task=asyncio.run_coroutine_threadsafe(tastFun(*args, **kwargs), loop=self._loop)
		return task.result()
	def __del__(self):
		log.info("loop close...")
		self._loop .close()
		log.info("loop close.")
	
		