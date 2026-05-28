import pytest
from sqlalchemy import Column, String, BigInteger, Integer, select
from sqlalchemy.orm import InstrumentedAttribute

from co6co_db_ext.po import BasePO
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.page_param import Page_param


class MockItemPO(BasePO):
    __tablename__ = "test_item"
    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    name = Column("name", String(64))
    category = Column("category", Integer)
    status = Column("status", Integer)


class MockFilterItems(absFilterItems):
    listSelectFields = [MockItemPO.id, MockItemPO.name]

    def __init__(self, po_type=MockItemPO):
        self.name = None
        self.category = None
       
        super().__init__(po_type)

    def filter(self):
        filters = []
        if self.name:
            filters.append(MockItemPO.name == self.name)
        if self.category is not None:
            filters.append(MockItemPO.category == self.category)
        return filters

    def getDefaultOrderBy(self):
        return [MockItemPO.id.asc()]


class TestAbsFilterItems:
    """测试 absFilterItems 抽象过滤器"""

    def test_init(self):
        """测试初始化"""
        filter_items = MockFilterItems(MockItemPO)
        assert filter_items.po_type == MockItemPO

    def test_offset_property(self):
        """测试 offset 属性（分页偏移量）"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageIndex = 1
        filter_items.pageSize = 10
        assert filter_items.offset == 0

        filter_items.pageIndex = 3
        filter_items.pageSize = 20
        assert filter_items.offset == 40

    def test_limit_property(self):
        """测试 limit 属性（每页数量）"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageSize = 25
        assert filter_items.limit == 25

    def test_getOrderBy_single_field_asc(self):
        """测试单字段升序排序"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "id"
        filter_items.order = "asc"

        order_by = filter_items.getOrderBy()
        assert len(order_by) == 1

    def test_getOrderBy_single_field_desc(self):
        """测试单字段降序排序"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "id"
        filter_items.order = "desc"

        order_by = filter_items.getOrderBy()
        assert len(order_by) == 1

    def test_getOrderBy_multiple_fields(self):
        """测试多字段排序"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "id,name"
        filter_items.order = "asc,desc"

        order_by = filter_items.getOrderBy()
        print("order_by",order_by)
        assert len(order_by) == 2

    def test_getOrderBy_mismatched_field_and_order_count(self):
        """测试字段和排序方向数量不匹配"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "id,name,status"
        filter_items.order = "asc"

        order_by = filter_items.getOrderBy()
        assert len(order_by) == 3

    def test_getOrderBy_empty_orderBy(self):
        """测试空排序字符串时使用默认排序"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = ""

        order_by = filter_items.getOrderBy()
        default_order = filter_items.getDefaultOrderBy()
        
        assert len(order_by) == len(default_order)
        for o, d in zip(order_by, default_order):
            assert str(o) == str(d)

    def test_getOrderBy_invalid_field(self):
        """测试无效字段名"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "invalid_field"
        filter_items.order = "asc"

        order_by = filter_items.getOrderBy()
        assert len(order_by) == 0

    def test_getOrderBy_partial_valid_fields(self):
        """测试部分有效字段"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.orderBy = "id,invalid_field"
        filter_items.order = "asc,desc"

        order_by = filter_items.getOrderBy()
        assert len(order_by) == 1

    def test_checkFieldValue_string(self):
        """测试字符串字段值检查"""
        filter_items = MockFilterItems(MockItemPO)

        assert filter_items.checkFieldValue("test") is True
        assert filter_items.checkFieldValue("") is False
        assert filter_items.checkFieldValue(None) is False

    def test_checkFieldValue_int(self):
        """测试整数字段值检查"""
        filter_items = MockFilterItems(MockItemPO)

        assert filter_items.checkFieldValue(0) is True
        assert filter_items.checkFieldValue(100) is True
        assert filter_items.checkFieldValue(-1) is True

    def test_checkFieldValue_bool(self):
        """测试布尔字段值检查"""
        filter_items = MockFilterItems(MockItemPO)

        assert filter_items.checkFieldValue(True) is True
        assert filter_items.checkFieldValue(False) is True

    def test_checkFieldValue_other_types(self):
        """测试其他类型值检查"""
        filter_items = MockFilterItems(MockItemPO)

        assert filter_items.checkFieldValue([]) is False
        assert filter_items.checkFieldValue({}) is False
        assert filter_items.checkFieldValue(1.5) is False


class TestAbsFilterItemsPagination:
    """测试 absFilterItems 分页功能"""

    def test_pagination_first_page(self):
        """测试第一页分页"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageIndex = 1
        filter_items.pageSize = 10

        assert filter_items.offset == 0
        assert filter_items.limit == 10

    def test_pagination_middle_page(self):
        """测试中间页分页"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageIndex = 5
        filter_items.pageSize = 20

        assert filter_items.offset == 80
        assert filter_items.limit == 20

    def test_pagination_last_page(self):
        """测试最后一页分页"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageIndex = 10
        filter_items.pageSize = 10

        assert filter_items.offset == 90
        assert filter_items.limit == 10


class TestAbsFilterItemsSelect:
    """测试 absFilterItems 查询构建功能"""

    def test_create_list_select_with_list_fields(self):
        """测试使用列表字段创建查询"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.name = "test"
        filter_items.category = 1

        select_stmt = filter_items.create_List_select()
        assert select_stmt is not None

    def test_list_select_with_pagination(self):
        """测试带分页的列表查询"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.pageIndex = 2
        filter_items.pageSize = 15
        filter_items.orderBy = "id"
        filter_items.order = "asc"


        select_stmt = filter_items.list_select
        assert select_stmt is not None

    def test_count_select(self):
        """测试计数查询"""
        filter_items = MockFilterItems(MockItemPO)
        filter_items.category = 1

        count_stmt = filter_items.count_select
        assert count_stmt is not None