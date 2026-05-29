import pytest
from multiprocessing import Process
import time
from co6co_biz.services.cacheManage import SharedCache, SharedMemoryCache


def _shared_cache_worker(cache, key, value):
    """SharedCache 多进程测试的工作函数"""
    cache.set(key, value)


def _shared_memory_worker(cache, key, value):
    """SharedMemoryCache 多进程测试的工作函数"""
    cache.set(key, value)


def _writer(cache, iterations):
    """并发写测试的工作函数"""
    for i in range(iterations):
        cache.set("counter", i)


def _reader(cache, iterations):
    """并发读测试的工作函数"""
    values = []
    for i in range(iterations):
        val = cache.get("counter")
        if val is not None:
            values.append(val)
    return values


class TestSharedCache:
    """测试 SharedCache 类"""

    def setup_method(self):
        """每个测试前的 setup"""
        self.cache = SharedCache()

    def teardown_method(self):
        """每个测试后的 teardown"""
        self.cache.clear()

    def test_init(self):
        """测试初始化"""
        cache = SharedCache()
        assert cache.cache is not None
        assert len(cache.cache) == 0

    def test_set_and_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_get_with_default(self):
        """测试获取不存在的键时返回默认值"""
        self.cache.set("existing", "value")
        result = self.cache.get("nonexistent", "default_value")
        assert result == "default_value"

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        self.cache.set("existing", "value")
        result = self.cache.get("nonexistent")
        assert result is None

    def test_set_complex_object(self):
        """测试存储复杂对象"""
        data = {
            "name": "test",
            "age": 25,
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        self.cache.set("complex", data)
        result = self.cache.get("complex")
        assert result == data

    def test_delete(self):
        """测试删除键"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        self.cache.delete("key1")
        assert self.cache.get("key1") is None

    def test_delete_nonexistent(self):
        """测试删除不存在的键"""
        self.cache.delete("nonexistent")
        assert True

    def test_clear(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        self.cache.clear()
        
        assert len(self.cache.cache) == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        assert self.cache.get("key3") is None

    def test_set_multiple_keys(self):
        """测试设置多个键值对"""
        for i in range(10):
            self.cache.set(f"key{i}", f"value{i}")
        
        for i in range(10):
            assert self.cache.get(f"key{i}") == f"value{i}"

    def test_overwrite_value(self):
        """测试覆盖已有键的值"""
        self.cache.set("key", "value1")
        assert self.cache.get("key") == "value1"
        
        self.cache.set("key", "value2")
        assert self.cache.get("key") == "value2"

    @pytest.mark.skip(reason="Windows 上 Manager 对象无法通过 spawn 方式跨进程传递")
    def test_multiprocess_access(self):
        """测试多进程访问"""
        self.cache.set("initial", "value")
        
        p1 = Process(target=_shared_cache_worker, args=(self.cache, "p1", "process1"))
        p2 = Process(target=_shared_cache_worker, args=(self.cache, "p2", "process2"))
        
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        assert self.cache.get("p1") == "process1"
        assert self.cache.get("p2") == "process2"
        assert self.cache.get("initial") == "value"


class TestSharedMemoryCache:
    """测试 SharedMemoryCache 类"""

    def setup_method(self):
        """每个测试前的 setup"""
        self.cache_name = f"test_cache_{time.time()}"
        self.cache = SharedMemoryCache(name=self.cache_name, size=1024 * 1024)

    def teardown_method(self):
        """每个测试后的 teardown"""
        try:
            self.cache.close()
        except Exception:
            pass

    def test_init(self):
        """测试初始化"""
        cache_name = f"test_init_{time.time()}"
        cache = SharedMemoryCache(name=cache_name, size=1024 * 1024)
        assert cache.name == cache_name
        assert cache.size == 1024 * 1024
        assert cache.shm is not None
        cache.close()

    def test_set_and_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_get_with_default(self):
        """测试获取不存在的键时返回默认值"""
        self.cache.set("existing", "value")
        result = self.cache.get("nonexistent", "default_value")
        assert result == "default_value"

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        self.cache.set("existing", "value")
        result = self.cache.get("nonexistent")
        assert result is None

    def test_set_complex_object(self):
        """测试存储复杂对象"""
        data = {
            "name": "test",
            "age": 25,
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        self.cache.set("complex", data)
        result = self.cache.get("complex")
        assert result == data

    def test_overwrite_value(self):
        """测试覆盖已有键的值"""
        self.cache.set("key", "value1")
        assert self.cache.get("key") == "value1"
        
        self.cache.set("key", "value2")
        assert self.cache.get("key") == "value2"

    def test_set_numeric_value(self):
        """测试存储数值"""
        self.cache.set("int", 42)
        assert self.cache.get("int") == 42
        
        self.cache.set("float", 3.14)
        assert self.cache.get("float") == 3.14

    def test_set_boolean_value(self):
        """测试存储布尔值"""
        self.cache.set("true_val", True)
        assert self.cache.get("true_val") is True
        
        self.cache.set("false_val", False)
        assert self.cache.get("false_val") is False

    def test_set_none_value(self):
        """测试存储 None 值"""
        self.cache.set("none_val", None)
        result = self.cache.get("none_val")
        assert result is None

    @pytest.mark.skip(reason="SharedMemoryCache 对象无法跨进程传递（Lock 和 SharedMemory 无法 pickle）")
    def test_multiprocess_access(self):
        """测试多进程访问"""
        self.cache.set("initial", "value")
        
        p1 = Process(target=_shared_memory_worker, args=(self.cache, "p1", "process1"))
        p2 = Process(target=_shared_memory_worker, args=(self.cache, "p2", "process2"))
        
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        assert self.cache.get("p1") == "process1"
        assert self.cache.get("p2") == "process2"
        assert self.cache.get("initial") == "value"

    def test_close(self):
        """测试关闭缓存"""
        cache_name = f"test_close_{time.time()}"
        cache = SharedMemoryCache(name=cache_name, size=1024 * 1024)
        cache.set("key", "value")
        
        cache.close()
        
        with pytest.raises(Exception):
            cache.set("key2", "value2")

    def test_large_data_error(self):
        """测试大数据错误"""
        cache_name = f"test_large_{time.time()}"
        small_cache = SharedMemoryCache(name=cache_name, size=10)
        
        with pytest.raises(ValueError):
            small_cache.set("key", "this is a very large string that exceeds the limit")
        
        small_cache.close()

    @pytest.mark.skip(reason="SharedMemoryCache 对象无法跨进程传递（Lock 和 SharedMemory 无法 pickle）")
    def test_concurrent_read_write(self):
        """测试并发读写"""
        iterations = 100
        
        p_write = Process(target=_writer, args=(self.cache, iterations))
        p_read = Process(target=_reader, args=(self.cache, iterations))
        
        p_write.start()
        p_read.start()
        p_write.join()
        p_read.join()
