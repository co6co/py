import multiprocessing
from multiprocessing.managers import BaseManager, DictProxy
import time

class SharedDataManager:
    """管理共享数据的类"""
    
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.shared_dict = self.manager.dict()
        self.shared_list = self.manager.list()
        self.lock = self.manager.Lock()
    
    def worker(self, worker_id):
        """工作进程函数"""
        try:
            with self.lock:
                self.shared_dict[f"worker_{worker_id}"] = {
                    "pid": multiprocessing.current_process().pid,
                    "time": time.time()
                }
                self.shared_list.append(f"Data from worker {worker_id}") 
            # 模拟工作
            time.sleep(0.5)
            
            return f"Worker {worker_id} completed"
            
        except Exception as e:
            return f"Worker {worker_id} error: {e}"
    
    def run_workers(self, num_workers=3):
        """运行工作进程"""
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(self.worker, range(num_workers))
        
        # 打印结果
        print("Results:", results)
        print("Shared dict:", dict(self.shared_dict))
        print("Shared list:", list(self.shared_list))

def manager_example():
    """使用 Manager 的示例"""
    manager = SharedDataManager()
    manager.run_workers(3)