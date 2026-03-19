"""
运行测试 python -m pytest tests/ -v
"""
import pytest # pip install pytest

# 参数化测试
@pytest.mark.parametrize("a, b, expected", [(1, 1, 2), (2, 3, 5), (0, 0, 0)])
def test_add(a, b, expected):
    assert a + b == expected
#@pytest.fixture
#def db_connection():
#    # 前置操作：建立数据库连接
#    conn = create_connection()
#    yield conn  # 返回连接对象给测试函数
#    # 后置操作：关闭数据库连接
#    conn.close()
#@pytest.mark.parametrize("user", [1, 2, 3], indirect=True)
#def test_user(user):
#    assert user is not None 
@pytest.fixture #提供测试所需的预设数据、环境或资源
def test_data():
    return {'key': 'value'}

def test_with_fixture(test_data):
    print("test_example2",test_data)
    assert test_data['key'] == 'value'

def test_add():
    """测试加法功能"""
    result = 1 + 1
    assert result == 2
