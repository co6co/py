# pipe_utils.py
import pickle
import struct
import zlib
from typing import Any
from multiprocessing.connection import PipeConnection

def send_large_data_safely(conn: PipeConnection, data: Any, chunk_size: int = 1024 * 1024) -> bool:
    """
    安全地发送大数据
    
    Args:
        conn: 管道连接
        data: 要发送的数据（必须可序列化）
        chunk_size: 块大小，默认1MB
        
    Returns:
        是否发送成功
    """
    try:
        # 1. 序列化数据
        serialized_data = pickle.dumps(data)
        
        # 2. 可选压缩
        compressed_data = zlib.compress(serialized_data, level=3)
        
        # 3. 发送元数据
        total_size = len(compressed_data)
        metadata = {
            "total_size": total_size,
            "chunk_size": chunk_size,
            "compressed": True,
            "version": "1.0"
        }
        conn.send(metadata)
        
        # 4. 分块发送
        for i in range(0, total_size, chunk_size):
            chunk = compressed_data[i:i+chunk_size]
            conn.send(chunk)
        
        # 5. 发送结束标记
        conn.send(b"END")
        
        return True
        
    except Exception as e:
        print(f"发送失败: {e}")
        # 发送错误信息
        try:
            conn.send({"error": str(e)})
        except:
            pass
        return False

def receive_large_data_chunked(conn: PipeConnection, timeout: float = 30.0) -> Any:
    """
    接收分块的大数据
    
    Args:
        conn: 管道连接
        timeout: 超时时间（秒）
        
    Returns:
        反序列化后的数据，或None（如果失败）
    """
    import time
    start_time = time.time()
    
    try:
        # 1. 接收元数据
        metadata = conn.recv()
        
        # 检查错误
        if isinstance(metadata, dict) and "error" in metadata:
            raise RuntimeError(f"发送端错误: {metadata['error']}")
        
        # 验证元数据
        required_keys = ["total_size", "chunk_size"]
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"元数据缺少必要字段: {key}")
        
        total_size = metadata["total_size"]
        chunk_size = metadata.get("chunk_size", 1024 * 1024)
        compressed = metadata.get("compressed", False)
        
        if total_size <= 0:
            raise ValueError(f"无效的总大小: {total_size}")
        
        # 2. 接收数据块
        chunks = []
        received_size = 0
        
        while received_size < total_size:
            # 检查超时
            if time.time() - start_time > timeout:
                raise TimeoutError(f"接收超时: {timeout}秒")
            
            chunk = conn.recv()
            
            # 检查是否是结束标记
            if chunk == b"END":
                break
            
            chunks.append(chunk)
            received_size += len(chunk)
        
        # 3. 合并数据
        combined_data = b"".join(chunks)
        
        if len(combined_data) != total_size:
            raise RuntimeError(
                f"数据不完整: 期望 {total_size} 字节, 收到 {len(combined_data)} 字节"
            )
        
        # 4. 解压缩（如果需要）
        if compressed:
            try:
                combined_data = zlib.decompress(combined_data)
            except zlib.error as e:
                raise RuntimeError(f"解压缩失败: {e}")
        
        # 5. 反序列化
        try:
            return pickle.loads(combined_data)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"反序列化失败: {e}")
        
    except EOFError:
        print("连接已关闭")
        return None
    except Exception as e:
        print(f"接收失败: {e}")
        return None