# test_pipe_utils.py
import unittest
import multiprocessing
import pickle
import zlib
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pipe_utils import send_large_data_safely, receive_large_data_chunked

class TestPipeUtils(unittest.TestCase):
    """单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
    
    def tearDown(self):
        """测试后清理"""
        self.parent_conn.close()
        self.child_conn.close()
    
    def test_serialization_deserialization(self):
        """测试基本序列化和反序列化"""
        test_data = {
            "string": "hello world",
            "number": 42,
            "list": [1, 2, 3, {"nested": "data"}],
            "none": None,
            "bool": True
        }
        
        # 测试序列化
        serialized = pickle.dumps(test_data)
        self.assertIsInstance(serialized, bytes)
        
        # 测试反序列化
        deserialized = pickle.loads(serialized)
        self.assertEqual(test_data, deserialized)
    
    def test_compression(self):
        """测试压缩解压缩"""
        # 创建重复数据，易于压缩
        original_data = b"test" * 1000
        
        # 压缩
        compressed = zlib.compress(original_data, level=3)
        self.assertLess(len(compressed), len(original_data))
        
        # 解压缩
        decompressed = zlib.decompress(compressed)
        self.assertEqual(original_data, decompressed)
    
    def test_metadata_format(self):
        """测试元数据格式"""
        with patch('pipe_utils.pickle.dumps') as mock_pickle:
            mock_pickle.return_value = b"test"
            
            with patch('pipe_utils.zlib.compress') as mock_compress:
                mock_compress.return_value = b"compressed"
                
                # 模拟发送
                mock_conn = Mock()
                send_large_data_safely(mock_conn, "test", chunk_size=1024)
                
                # 验证元数据格式
                metadata_call = mock_conn.send.call_args_list[0]
                metadata = metadata_call[0][0]
                
                self.assertIn("total_size", metadata)
                self.assertIn("chunk_size", metadata)
                self.assertIn("compressed", metadata)
                self.assertEqual(metadata["chunk_size"], 1024)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试发送不可序列化的数据
        class Unpicklable:
            def __reduce__(self):
                raise pickle.PicklingError("Cannot pickle")
        
        result = send_large_data_safely(self.parent_conn, Unpicklable())
        self.assertFalse(result)
    
    def test_chunking_logic(self):
        """测试分块逻辑"""
        data = b"x" * 5000  # 5KB数据
        
        with patch('pipe_utils.pickle.dumps') as mock_pickle:
            mock_pickle.return_value = data
            
            with patch('pipe_utils.zlib.compress') as mock_compress:
                mock_compress.return_value = data
                
                mock_conn = Mock()
                send_large_data_safely(mock_conn, data, chunk_size=1024)
                
                # 验证分块次数
                # 1次元数据 + 5个数据块 + 1次结束标记
                self.assertEqual(mock_conn.send.call_count, 7)
    
    def test_receive_timeout(self):
        """测试接收超时"""
        with patch('time.time', side_effect=[0, 31]):  # 模拟超时
            metadata = {"total_size": 1000, "chunk_size": 1024}
            
            mock_conn = Mock()
            mock_conn.recv.return_value = metadata
            
            result = receive_large_data_chunked(mock_conn, timeout=30)
            self.assertIsNone(result)
    
    def test_invalid_metadata(self):
        """测试无效元数据"""
        # 缺少必要字段
        invalid_metadata = {"chunk_size": 1024}  # 缺少 total_size
        
        mock_conn = Mock()
        mock_conn.recv.return_value = invalid_metadata
        
        result = receive_large_data_chunked(mock_conn)
        self.assertIsNone(result)
    
    def test_receive_incomplete_data(self):
        """测试接收不完整数据"""
        metadata = {"total_size": 1000, "chunk_size": 1024, "compressed": False}
        
        mock_conn = Mock()
        mock_conn.recv.side_effect = [
            metadata,  # 元数据
            b"x" * 500,  # 只收到500字节
            b"END"  # 提前结束
        ]
        
        result = receive_large_data_chunked(mock_conn)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main(verbosity=2)