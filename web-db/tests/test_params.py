"""Tests for model.params module"""
import pytest
from co6co_web_db.model.params import associationParam


class TestAssociationParam:
    """Test cases for associationParam class"""

    def test_default_values_are_empty_lists(self):
        """Test default values are empty lists"""
        param = associationParam()
        assert param.add == []
        assert param.remove == []

    def test_can_set_add_attribute(self):
        """Test can set add attribute"""
        param = associationParam()
        param.add = [1, 2, 3]
        assert param.add == [1, 2, 3]

    def test_can_set_remove_attribute(self):
        """Test can set remove attribute"""
        param = associationParam()
        param.remove = [4, 5, 6]
        assert param.remove == [4, 5, 6]

    def test_multiple_operations(self):
        """Test multiple add and remove operations"""
        param = associationParam()
        param.add = [1, 2]
        param.remove = [3, 4]
        assert param.add == [1, 2]
        assert param.remove == [3, 4]
        param.add.append(5)
        assert param.add == [1, 2, 5]
        param.remove.pop()
        assert param.remove == [3]