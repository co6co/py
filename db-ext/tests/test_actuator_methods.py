import pytest
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.engine.row import Row, RowMapping
from sqlalchemy import Column, Integer, String
from co6co_db_ext.actuator import Actuator
from co6co_db_ext.po import BasePO


class MockPO(BasePO):
    __tablename__ = "mock_table"
    id = Column(Integer, primary_key=True)
    name = Column(String(64))


class SimpleDictClass:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestActuatorStaticMethods:
    """测试 Actuator 静态方法"""

    def test_remove_db_instance_state_from_list(self):
        """测试从列表中移除 SQLAlchemy 实例状态"""
        mock_po1 = SimpleDictClass(
            id=1, name="test1", _sa_instance_state=MagicMock())
        mock_po2 = SimpleDictClass(
            id=2, name="test2", _sa_instance_state=MagicMock())

        result = Actuator.remove_db_instance_state([mock_po1, mock_po2])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"id": 1, "name": "test1"}
        assert result[1] == {"id": 2, "name": "test2"}

    def test_remove_db_instance_state_from_single_object(self):
        """测试从单个对象移除 SQLAlchemy 实例状态"""
        mock_po = SimpleDictClass(
            id=1, name="test", _sa_instance_state=MagicMock())

        result = Actuator.remove_db_instance_state(mock_po)

        assert result == {"id": 1, "name": "test"}

    def test_remove_db_instance_state_nested_po(self):
        """测试嵌套 PO 对象的实例状态移除"""
        mock_inner = SimpleDictClass(
            id=10, name="inner", _sa_instance_state=MagicMock())
        mock_outer = SimpleDictClass(
            id=1, child=mock_inner, _sa_instance_state=MagicMock())

        result = Actuator.remove_db_instance_state([mock_outer])
        print("XXXXXXXXX", result[0], result)

        assert isinstance(result, list)
        assert result[0]["id"] == 1
        assert result[0]["child"]["id"] == 10

    def test_remove_db_instance_state_non_iterable(self):
        """测试非可迭代对象"""
        result = Actuator.remove_db_instance_state("string")
        assert result == "string"

        result = Actuator.remove_db_instance_state(123)
        assert result == 123

    def test_remove_db_instance_state_empty_list(self):
        """测试空列表"""
        result = Actuator.remove_db_instance_state([])
        assert result == []

    def test_row2dict_basic(self):
        """测试 Row 转字典"""
        mock_row = MagicMock(spec=Row)
        mock_row._fields = ("id", "name")
        mock_row.__getitem__ = lambda self, i: [1, "test"][i]
        mock_row.__len__ = lambda self: 2

        result = Actuator.row2dict(mock_row)

        assert result == {"id": 1, "name": "test"}

    def test_row2dict_with_po_instance(self):
        """测试包含 PO 实例的 Row 转字典"""
        mock_po = SimpleDictClass(id=10, _sa_instance_state=MagicMock())

        mock_row = MagicMock()
        mock_row._fields = ("id", "po")
        mock_row.__getitem__ = lambda self, i: [1, mock_po][i]
        mock_row.__len__ = lambda self: 2
        print("长度",mock_row[1],mock_row._fields[1])
        result = Actuator.row2dict(mock_row)
        print(result)
        assert result["id"] == 1
        assert "po" in result

    def test_one2Dict_row(self):
        """测试 Row 转字典"""
        mock_row = MagicMock(spec=Row)
        mock_row._fields = ("id", "name")
        mock_row.__iter__ = lambda self: iter([1, "test"])

        result = Actuator.one2Dict(mock_row)
        assert isinstance(result, dict)
        assert result == {"id": 1, "name": "test"}

    def test_one2Dict_row_mapping(self):
        """测试 RowMapping 转字典"""
        mock_mapping = {"id": 1, "name": "test"}

        result = Actuator.one2Dict(mock_mapping)
        assert result == {"id": 1, "name": "test"}

    def test_one2Dict_dict(self):
        """测试普通字典直接返回"""
        mock_dict = {"id": 1, "name": "test"}

        result = Actuator.one2Dict(mock_dict)

        assert result == {"id": 1, "name": "test"}

    def test_list2Dict(self):
        """测试 Row 列表转字典列表"""
        mock_row1 = MagicMock(spec=Row)
        mock_row1._fields = ("id", "name")
        mock_row1.__iter__ = lambda self: iter([(1, "test1")])
        mock_row1.__getitem__ = lambda self, i: {
            "id": 1, "name": "test1"}.get(i)

        mock_row2 = MagicMock(spec=Row)
        mock_row2._fields = ("id", "name")
        mock_row2.__iter__ = lambda self: iter([(2, "test2")])
        mock_row2.__getitem__ = lambda self, i: {
            "id": 2, "name": "test2"}.get(i)

        result = Actuator.list2Dict([mock_row1, mock_row2])

        assert isinstance(result, list)
        assert len(result) == 2

    def test_select_result_strategy_entity(self):
        """测试实体查询的策略选择"""
        from sqlalchemy.orm import Mapper

        mock_mapper = MagicMock(spec=Mapper)
        mock_select = MagicMock()
        mock_select._raw_columns = [mock_mapper]

        result = Actuator.select_result_strategy(mock_select)

        assert result == "scalars"

    def test_select_result_strategy_single_column(self):
        """测试单列查询的策略选择"""
        from sqlalchemy import Column

        mock_column = MagicMock(spec=Column)
        mock_select = MagicMock()
        mock_select._raw_columns = [mock_column]

        result = Actuator.select_result_strategy(mock_select)

        assert result == "scalar"

    def test_select_result_strategy_multi_column(self):
        """测试多列查询的策略选择"""
        from sqlalchemy import Column

        mock_col1 = MagicMock(spec=Column)
        mock_col2 = MagicMock(spec=Column)
        mock_select = MagicMock()
        mock_select._raw_columns = [mock_col1, mock_col2]

        result = Actuator.select_result_strategy(mock_select)

        assert result == "mappings"

    def test_is_entity_select_true(self):
        """测试实体查询判断 - 真"""
        from .right import UserPO
        from sqlalchemy import Select 
        select=Select(UserPO).filter(UserPO.id>0)
        
        result = Actuator.is_entity_select(select)

        assert result is True

    def test_is_entity_select_false(self):
        """测试实体查询判断 - 假"""
        
        from .right import UserPO
        from sqlalchemy import Select 
        select=Select(UserPO.id,UserPO.userName).filter(UserPO.id>0) 
        result = Actuator.is_entity_select(select)

        assert result is False
