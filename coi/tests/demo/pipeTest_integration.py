# test_integration.py
import multiprocessing
import time
import pickle
import hashlib
import tempfile
import os
from pathlib import Path
import numpy as np
from pipe_utils import send_large_data_safely, receive_large_data_chunked

def sender_process(conn, data, chunk_size, result_queue):
    """发送进程"""
    try:
        success = send_large_data_safely(conn, data, chunk_size=chunk_size)
        result_queue.put({"success": success, "role": "sender"})
    except Exception as e:
        result_queue.put({"success": False, "error": str(e), "role": "sender"})
    finally:
        conn.close()

def receiver_process(conn, result_queue, timeout=30):
    """接收进程"""
    try:
        received_data = receive_large_data_chunked(conn, timeout=timeout)
        result_queue.put({
            "success": received_data is not None,
            "data": received_data,
            "role": "receiver"
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
            "role": "receiver"
        })
    finally:
        conn.close()

def test_end_to_end():
    """端到端集成测试"""
    print("=" * 60)
    print("端到端集成测试")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "小数据（小于块大小）",
            "data": {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}},
            "chunk_size": 1024
        },
        {
            "name": "中等数据（多个块）",
            "data": b"x" * (3 * 1024 * 1024),  # 3MB
            "chunk_size": 1024 * 1024  # 1MB块
        },
        {
            "name": "大数据（压缩效果好）",
            "data": b"repeat" * 100000,  # 约2MB
            "chunk_size": 1024 * 1024
        },
        {
            "name": "复杂嵌套结构",
            "data": {
                "users": [
                    {"id": i, "name": f"user_{i}", "data": list(range(100))}
                    for i in range(100)
                ],
                "metadata": {
                    "count": 100,
                    "timestamp": time.time(),
                    "active": True
                }
            },
            "chunk_size": 1024
        },
        {
            "name": "NumPy数组",
            "data": np.random.rand(1000, 1000),  # 约8MB
            "chunk_size": 1024 * 1024
        },
        {
            "name": "混合数据类型",
            "data": {
                "text": "Hello, World! " * 1000,
                "binary": b"\x00\x01\x02\x03" * 1000,
                "numbers": [1.5, 2.7, 3.14],
                "matrix": np.eye(100).tolist(),
                "none": None,
                "boolean": [True, False, True]
            },
            "chunk_size": 1024
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print("-" * 40)
        
        # 创建管道和队列
        parent_conn, child_conn = multiprocessing.Pipe()
        result_queue = multiprocessing.Queue()
        
        # 计算原始数据哈希
        original_hash = hashlib.md5(pickle.dumps(test_case["data"])).hexdigest()
        
        # 创建进程
        sender = multiprocessing.Process(
            target=sender_process,
            args=(parent_conn, test_case["data"], test_case["chunk_size"], result_queue)
        )
        receiver = multiprocessing.Process(
            target=receiver_process,
            args=(child_conn, result_queue, 30)
        )
        
        # 启动并等待进程
        start_time = time.time()
        
        sender.start()
        receiver.start()
        
        sender.join(timeout=60)
        receiver.join(timeout=60)
        
        end_time = time.time()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 分析结果
        sender_result = next((r for r in results if r.get("role") == "sender"), None)
        receiver_result = next((r for r in results if r.get("role") == "receiver"), None)
        
        if sender_result and receiver_result:
            if sender_result["success"] and receiver_result["success"]:
                # 验证数据完整性
                received_data = receiver_result["data"]
                received_hash = hashlib.md5(pickle.dumps(received_data)).hexdigest()
                
                if original_hash == received_hash:
                    elapsed = end_time - start_time
                    data_size = len(pickle.dumps(test_case["data"]))
                    print(f"✅ 测试通过!")
                    print(f"   数据大小: {data_size:,} 字节")
                    print(f"   耗时: {elapsed:.2f} 秒")
                    print(f"   速度: {data_size / elapsed / 1024 / 1024:.2f} MB/s")
                else:
                    print(f"❌ 数据损坏!")
                    all_passed = False
            else:
                print(f"❌ 失败!")
                if "error" in sender_result:
                    print(f"   发送端错误: {sender_result['error']}")
                if "error" in receiver_result:
                    print(f"   接收端错误: {receiver_result['error']}")
                all_passed = False
        else:
            print(f"❌ 进程异常!")
            all_passed = False
        
        # 清理
        if sender.is_alive():
            sender.terminate()
        if receiver.is_alive():
            receiver.terminate()
    
    print(f"\n" + "=" * 60)
    if all_passed:
        print("✅ 所有端到端测试通过!")
    else:
        print("❌ 部分测试失败!")
    return all_passed

def test_stress():
    """压力测试"""
    print("\n" + "=" * 60)
    print("压力测试")
    print("=" * 60)
    
    def stress_sender(conn, num_messages, data_size):
        """发送大量消息"""
        for i in range(num_messages):
            data = {"id": i, "data": b"x" * data_size, "timestamp": time.time()}
            send_large_data_safely(conn, data, chunk_size=1024)
        conn.send("DONE")
    
    def stress_receiver(conn, result_queue):
        """接收大量消息"""
        count = 0
        while True:
            try:
                data = receive_large_data_chunked(conn, timeout=5)
                if data == "DONE":
                    break
                if data is not None:
                    count += 1
            except:
                break
        result_queue.put(count)
    
    # 测试参数
    test_configs = [
        {"messages": 10, "size": 1000},      # 10条小消息
        {"messages": 100, "size": 10000},    # 100条中消息
        {"messages": 1000, "size": 100},     # 1000条小消息
    ]
    
    for config in test_configs:
        print(f"\n测试: {config['messages']} 条消息, 每条 {config['size']} 字节")
        
        parent_conn, child_conn = multiprocessing.Pipe()
        result_queue = multiprocessing.Queue()
        
        sender = multiprocessing.Process(
            target=stress_sender,
            args=(parent_conn, config["messages"], config["size"])
        )
        receiver = multiprocessing.Process(
            target=stress_receiver,
            args=(child_conn, result_queue)
        )
        
        start_time = time.time()
        
        sender.start()
        receiver.start()
        
        sender.join(timeout=30)
        receiver.join(timeout=30)
        
        elapsed = time.time() - start_time
        
        received_count = result_queue.get() if not result_queue.empty() else 0
        
        if received_count == config["messages"]:
            print(f"✅ 通过! 收到 {received_count} 条消息, 耗时 {elapsed:.2f} 秒")
            print(f"   吞吐量: {config['messages'] / elapsed:.2f} 消息/秒")
        else:
            print(f"❌ 失败! 期望 {config['messages']}, 收到 {received_count}")
        
        sender.terminate()
        receiver.terminate()

def test_concurrent():
    """并发测试"""
    print("\n" + "=" * 60)
    print("并发测试")
    print("=" * 60)
    
    def concurrent_worker(worker_id, pipe, data, result_queue):
        """并发工作线程"""
        try:
            # 随机选择发送或接收
            import random
            if random.choice([True, False]):
                # 发送
                success = send_large_data_safely(pipe, data)
                result_queue.put({"worker": worker_id, "action": "send", "success": success})
            else:
                # 尝试接收
                received = receive_large_data_chunked(pipe, timeout=1)
                result_queue.put({"worker": worker_id, "action": "receive", "success": received is not None})
        except Exception as e:
            result_queue.put({"worker": worker_id, "error": str(e)})
    
    # 创建多个管道
    num_workers = 5
    pipes = [multiprocessing.Pipe() for _ in range(num_workers)]
    result_queue = multiprocessing.Queue()
    
    test_data = {"test": "concurrent", "value": 42}
    
    workers = []
    for i in range(num_workers):
        parent_conn, child_conn = pipes[i]
        worker = multiprocessing.Process(
            target=concurrent_worker,
            args=(i, parent_conn, test_data, result_queue)
        )
        workers.append(worker)
    
    # 启动所有worker
    for worker in workers:
        worker.start()
    
    # 等待
    for worker in workers:
        worker.join(timeout=10)
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f"并发测试结果:")
    for result in results:
        if "error" in result:
            print(f"  Worker {result['worker']}: ❌ {result['error']}")
        else:
            print(f"  Worker {result['worker']}: ✅ {result['action']} - {result['success']}")
    
    # 清理
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    
    for parent_conn, child_conn in pipes:
        parent_conn.close()
        child_conn.close()

if __name__ == "__main__":
    # 运行集成测试
    test_end_to_end()
    
    # 运行压力测试
    test_stress()
    
    # 运行并发测试
    test_concurrent()