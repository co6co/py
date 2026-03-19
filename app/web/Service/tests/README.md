# Python测试指南

## 测试框架

### 1. unittest (Python标准库)
- **优点**：内置库，无需安装
- **缺点**：语法较繁琐

### 2. pytest
- **优点**：语法简洁，功能强大
- **安装**：`pip install pytest`

## 运行测试

### 使用unittest运行
```bash
python -m unittest tests/test_example.py
```

### 使用pytest运行
```bash
python -m pytest tests/ -v  # 有更多的参数但不会显示 print
python -m pytest test/ -s   # 禁止输出捕获
python -m pytest test/ -s -V # 即可看到测试信息又能看到 print 输出
python -m pytest test/ -rA # -r 显示测试的额外信息，包含通过的测试

```

## 测试编写最佳实践

### 1. 测试命名规范
- 测试类名：`TestXXX`
- 测试方法名：`test_xxx`
- 文件名：`test_xxx.py`

### 2. 测试结构
- **Arrange**：准备测试数据和环境
- **Act**：执行被测试的代码
- **Assert**：验证结果是否符合预期

### 3. 测试类型
- **单元测试**：测试单个函数或方法
- **集成测试**：测试多个组件的交互
- **端到端测试**：测试整个系统的流程

### 4. 测试覆盖
- **语句覆盖**：确保每一行代码都被执行
- **分支覆盖**：确保每一个条件分支都被测试
- **路径覆盖**：确保每一个可能的执行路径都被测试

### 5. 测试隔离
- 每个测试应该独立运行
- 测试之间不应该有依赖关系
- 使用`setUp`和`tearDown`方法来设置和清理测试环境

### 6. 测试数据
- 使用明确的测试数据
- 包含正常情况、边界情况和异常情况
- 避免使用真实的外部依赖（如数据库、网络连接），可以使用mock

### 7. 测试文档
- 为测试方法添加文档字符串，说明测试的目的
- 为测试文件添加说明，说明测试的范围和方法

## 示例测试文件

### unittest风格
```python
import unittest

class TestExample(unittest.TestCase):
    def test_add(self):
        """测试加法功能"""
        result = 1 + 1
        self.assertEqual(result, 2)

if __name__ == '__main__':
    unittest.main()
```

### pytest风格
```python
def test_add():
    """测试加法功能"""
    result = 1 + 1
    assert result == 2
```

## 高级特性

### 1. Mock
使用`unittest.mock`来模拟外部依赖：

```python
from unittest.mock import patch

@patch('module.function')
def test_with_mock(mock_function):
    mock_function.return_value = 'mocked result'
    # 测试代码
```

### 2. 参数化测试
使用pytest的`@pytest.mark.parametrize`：

```python
import pytest

@pytest.mark.parametrize("a, b, expected", [(1, 1, 2), (2, 3, 5), (0, 0, 0)])
def test_add(a, b, expected):
    assert a + b == expected
```

### 3. 测试夹具（Fixtures）
使用pytest的fixture来设置测试环境：

```python
import pytest

@pytest.fixture
def test_data():
    return {"key": "value"}

def test_with_fixture(test_data):
    assert test_data["key"] == "value"
```
