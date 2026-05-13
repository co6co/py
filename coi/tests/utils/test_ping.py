import asyncio,pytest

from co6co.utils.network import async_ping
from co6co.utils. aio import AsyncLimiter
import time

@pytest.fixture
def hosts():
    hosts = [
        "8.8.8.8",
        "1.1.1.1", 
        "192.168.1.150",
        "127.0.0.1",
        "192.168.1.254", 
        "baidu.com",
        "google.com",
    ]
    return hosts
limiter_g = AsyncLimiter(2) 
@limiter_g.wrap 
async def ping_limiter_warp(host,resultType :int|bool, ):
    return await async_ping(host,2,2,resultType)

async def ping_limiter_all(hosts,resultType :int|bool ):
    tasks = [ping_limiter_warp(host,resultType) for host in hosts]
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    return results,end_time-start_time
async def ping_bounded(hosts,resultType :int|bool,useLimiter :bool = True):
    if useLimiter:
        limiter = AsyncLimiter(2)   
        tasks = [limiter.run(async_ping(host,2,2,resultType)) for host in hosts]
    else:
        tasks = [async_ping(host,2,2,resultType) for host in hosts]
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    return results,end_time-start_time
async def ping_bounded_timeout(hosts,resultType :int|bool,*args ):
    
    time0=1.0
    time1=None
    print(f"time limit {time0} seconds，task timeout {time1} seconds")
    limiter = AsyncLimiter(2)   
    tasks = [limiter.run_timeout(async_ping(host,2,2,resultType),timeout=time0,task_timeout=time1) for host in hosts] 
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    print("data", results)
    end_time = time.time()
    return results,end_time-start_time

def show_result(hosts,results,resultType :int|bool,isAssert :bool = True): 
    for host, alive in zip(hosts, results):
        if isAssert:
            assert isinstance(alive,resultType)
        if isinstance(alive,bool):
            print(f"{host:15} -> {'UP' if alive else 'DOWN'}")
        elif isinstance(alive,float):
            print(f"{host:15} -> {f'{alive}' if alive else 'DOWN'}")
        else:
            print(f"{host:15} ->   'DOWN' ")
    pass
async def pingAll(hosts):
    params=[(hosts,bool,True),(hosts,float,True),(hosts,bool,False),(hosts,float,False),(hosts,bool,False),(hosts,float,False)]
    exec=[ping_bounded_timeout,ping_bounded] #,
    params=[(hosts,bool,True),(hosts,float,True)]
    
    for exc in exec:  
        for ps in params: 
            hosts,resultType,useLimiter = ps
            print(f"\n\nping {len(hosts)} hosts 结果类型： {resultType.__name__} 是否使用限速： {useLimiter} ")
            results,use_time  = await exc(*ps)
            print(f"ping 结果 {len(hosts)} hosts cost {use_time} seconds")
            show_result(hosts,results,resultType,isAssert=(exc!=ping_bounded_timeout))
    

    print("\n\n\n装饰器测试")
    params=[(hosts,bool),(hosts,float)]
    for ps in params: 
        hosts,resultType = ps
        results,use_time  = await ping_limiter_all(hosts,resultType)
        print(f"ping 结果 {len(hosts)} hosts cost {use_time} seconds")
        show_result(hosts,results,resultType)   
    
                
def test_ping(hosts):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(pingAll(hosts))
    loop.close()