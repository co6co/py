

self._semaphore = asyncio.Semaphore(5)  # 最多同时运行5个任务

async def run_task(self, taskId: str, t: AbsTaskMgr, config: Dict[str, Any]):
    async with self._semaphore:


for a in range(1,1000)
    # 让出事件循环控制权，允许其他任务执行
    await asyncio.sleep(0)
    print(a)
    

# 同步代码转异步
# 方案一
async def  _handler_(self,id: str ):
     # 将同步C SDK调用移到线程池
    result = await asyncio.to_thread(self._do, id)
    return result 
# 方案二 使用loop
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(
    self._executor,  # 自定义线程池
    self._do,
    id
)
# 方案三
for i in range(batch_count):
    # ... 原有逻辑
    await asyncio.sleep(0)  # 让出事件循环