

from cacheout import Cache 
import time
from datetime import datetime
cache = Cache(maxsize=256, ttl=10, timer=time.time, default=None) 

cache.set("123",datetime.now())
cache.set("99",datetime.now())

while True:
    ks=cache.keys()
    for k in ks:
        #v=cache.get(k)
        v="" 
        print(f"{k}:{v}") 

    time.sleep(1)