from multiprocessing import Manager
from multiprocessing import shared_memory, Lock
import pickle
from co6co.utils import log


class SharedCache:
    """
    跨进程通行
    读写加锁，执行缓慢
    存对象是副本，嵌套修改不同步
    shared_dict['a']['b'] = 123  # 无效！不同步！
    需要取出来，修改完后， 再赋值回去
    tmp = shared_dict['a']
    tmp['b'] = 123
    shared_dict['a'] = tmp  # 重新赋值才同步
    """ 
    def __init__(self):
        self.manager = Manager()
        self.cache = self.manager.dict()  # DictProxy

    # 存：必须整体赋值，确保同步
    def set(self, key, value):
        self.cache[key] = value

    # 取
    def get(self, key, default=None):
        return self.cache.get(key, default)

    # 删除
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

    # 清空
    def clear(self):
        self.cache.clear()


class SharedMemoryCache:
    """
    # 纯内存缓存（多进程安全、真正共享）
    def task(cache):
        cache.set("name", "小明")
        print(cache.get("name"))  # 多进程都能拿到

    if __name__ == '__main__':
        cache = SharedMemoryCache()
        p = Process(target=task, args=(cache,))
        p.start()
        p.join()
        print(cache.get("name"))  # 主进程也能读到！
    """

    def __init__(self, name="cache", size=1024 * 1024):
        self.name = name
        self.size = size
        self.lock = Lock()
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)

    def set(self, key, value):
        with self.lock:
            data = pickle.dumps({key: value})
            if len(data) > self.size:
                raise ValueError("数据太大，内存不足")
            self.shm.buf[: len(data)] = data

    def get(self, key, default=None):
        with self.lock:
            try:
                data = bytes(self.shm.buf).rstrip(b"\x00")
                cache = pickle.loads(data)
                return cache.get(key, default)
            except pickle.PickleError as e:
                log.error("pickle error: {}".format(e))

                return default

    def close(self):
        self.shm.close()
        self.shm.unlink()
