import pytest
from co6co_db_ext.page_param import Page_param


class TestPageParam:
    """测试 Page_param 分页参数类"""

    def test_init_default_values(self):
        """测试默认初始化值"""
        param = Page_param()
        assert param.pageIndex == 1
        assert param.pageSize == 10
        assert param.orderBy == ""
        assert param.order == "asc"

    def test_init_with_custom_values(self):
        """测试自定义初始化值"""
        param = Page_param()
        param.pageIndex = 5
        param.pageSize = 20
        param.orderBy = "id,name"
        param.order = "desc"

        assert param.pageIndex == 5
        assert param.pageSize == 20
        assert param.orderBy == "id,name"
        assert param.order == "desc"

    def test_get_db_page_index(self):
        """测试获取数据库页码索引（从0开始）"""
        param = Page_param()

        param.pageIndex = 1
        assert param.get_db_page_index() == 0

        param.pageIndex = 5
        assert param.get_db_page_index() == 4

        param.pageIndex = 10
        assert param.get_db_page_index() == 9

    def test_getMaxPageIndex_basic(self):
        """测试获取最大页码索引"""
        param = Page_param()
        param.pageSize = 10

        assert param.getMaxPageIndex(100) == 10
        assert param.getMaxPageIndex(101) == 11
        assert param.getMaxPageIndex(99) == 10
        assert param.getMaxPageIndex(0) == 0
        assert param.getMaxPageIndex(1) == 1

    def test_getMaxPageIndex_different_page_sizes(self):
        """测试不同页面大小下的最大页码"""
        param = Page_param()

        param.pageSize = 20
        assert param.getMaxPageIndex(100) == 5
        assert param.getMaxPageIndex(101) == 6

        param.pageSize = 50
        assert param.getMaxPageIndex(100) == 2
        assert param.getMaxPageIndex(101) == 3

        param.pageSize = 100
        assert param.getMaxPageIndex(50) == 1
        assert param.getMaxPageIndex(100) == 1
        assert param.getMaxPageIndex(101) == 2

    def test_edge_cases(self):
        """测试边界情况"""
        param = Page_param()
        param.pageSize = 10

        assert param.getMaxPageIndex(0) == 0
        assert param.getMaxPageIndex(-1) == 0

    def test_large_record_count(self):
        """测试大记录数"""
        param = Page_param()
        param.pageSize = 10

        assert param.getMaxPageIndex(1000000) == 100000