import multiprocessing
import array
import mmap

def send_with_shared_memory():
    """使用共享内存传输大数组"""
    
    def worker(shm_name, shape, dtype):
        """工作进程"""
        import numpy as np
        from multiprocessing import shared_memory
        
        # 连接到现有的共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        
        # 从共享内存创建 numpy 数组
        np_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # 使用数据
        result = np_array.mean()
        
        # 清理
        existing_shm.close()
        
        return result
    
    # 创建共享内存
    import numpy as np
    from multiprocessing import shared_memory
    
    # 创建大数组
    large_array = np.random.rand(10000, 10000)  # 800MB
    
    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=large_array.nbytes)
    
    # 将数据复制到共享内存
    shm_array = np.ndarray(large_array.shape, dtype=large_array.dtype, buffer=shm.buf)
    np.copyto(shm_array, large_array)
    
    # 启动进程
    p = multiprocessing.Process(
        target=worker,
        args=(shm.name, large_array.shape, large_array.dtype)
    )
    p.start()
    p.join()
    
    # 清理
    shm.close()
    shm.unlink()