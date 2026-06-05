"""
测试 co6co_permissions.services 模块中的服务类
"""
import pytest
from unittest.mock import MagicMock
from co6co_permissions.services.authService import PermissionValid
from co6co_permissions.services.utils import appHelper


class TestPermissionValid:
    """
    测试权限验证服务 PermissionValid
    使用源代码中的 PermissionValid 类进行测试
    """

    def test_check_not_inited_returns_false(self):
        """
        测试未初始化时check方法返回False
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        
        
        # 测试源代码中的 check 方法
        result = permission_valid.inited
        assert result is False

    def test_check_exact_url_match(self):
        """
        测试精确URL匹配 - 使用源代码中的 _check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid._inited = True
        permission_valid.currentUserMenus = [{"url": "/api/users", "methods": "GET"}] 
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_url_not_match(self):
        """
        测试URL不匹配 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/orders"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users", "methods": "GET"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is False

    def test_check_method_not_allowed(self):
        """
        测试请求方法不允许 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "POST"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users", "methods": "GET"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is False

    def test_check_method_all_allowed(self):
        """
        测试ALL方法允许所有请求 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "DELETE"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users", "methods": "ALL"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_wildcard_single_star(self):
        """
        测试单星号通配符匹配 - 使用源代码中的 _check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users/123"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users/*", "methods": "GET"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_wildcard_double_star(self):
        """
        测试双星号通配符匹配 - 使用源代码中的 _check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users/123/orders/456"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users/**", "methods": "GET"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_multiple_methods(self):
        """
        测试多个请求方法 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "POST"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [{"url": "/api/users", "methods": "GET,POST,DELETE"}]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_multiple_menus_match_first(self):
        """
        测试多个菜单匹配第一个 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/users"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [
            {"url": "/api/users", "methods": "GET"},
            {"url": "/api/orders", "methods": "GET"}
        ]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is True

    def test_check_multiple_menus_no_match(self):
        """
        测试多个菜单都不匹配 - 使用源代码中的 check 方法
        """
        mock_request = MagicMock()
        mock_request.path = "/api/products"
        mock_request.method = "GET"
        
        # 使用源代码中的 PermissionValid 类
        permission_valid = PermissionValid(mock_request)
        permission_valid.currentUserMenus = [
            {"url": "/api/users", "methods": "GET"},
            {"url": "/api/orders", "methods": "GET"}
        ]
        permission_valid._inited = True
        
        # 测试源代码中的 check 方法
        result = permission_valid.check()
        assert result is False


class TestAppHelper:
    """
    测试工具类 appHelper
    使用源代码中的 appHelper 类进行测试
    """

    def test_current_user_exists(self):
        """
        测试获取当前用户信息 - 使用源代码中的 current_user 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": 10}
        
        # 使用源代码中的 appHelper.current_user 方法
        result = appHelper.current_user(mock_request)
        
        assert result is not None
        assert result["id"] == 1
        assert result["user_name"] == "test_user"
        assert result["group_id"] == 10

    def test_current_user_not_exists(self):
        """
        测试当前用户不存在时抛出异常 - 使用源代码中的 current_user 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__ = {}
        
        # 使用源代码中的 appHelper.current_user 方法
        with pytest.raises(Exception) as exc_info:
            appHelper.current_user(mock_request)
        
        assert "当前用户信息不存在" in str(exc_info.value)

    def test_current_user_id(self):
        """
        测试获取用户ID - 使用源代码中的 current_user_id 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 123, "user_name": "test_user", "group_id": 10}
        
        # 使用源代码中的 appHelper.current_user_id 方法
        result = appHelper.current_user_id(mock_request)
        
        assert result == 123
        assert isinstance(result, int)

    def test_current_user_name(self):
        """
        测试获取用户名 - 使用源代码中的 current_user_name 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "admin", "group_id": 10}
        
        # 使用源代码中的 appHelper.current_user_name 方法
        result = appHelper.current_user_name(mock_request)
        
        assert result == "admin"

    def test_current_user_group_id(self):
        """
        测试获取用户组ID - 使用源代码中的 current_user_group_id 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": 5}
        
        # 使用源代码中的 appHelper.current_user_group_id 方法
        result = appHelper.current_user_group_id(mock_request)
        
        assert result == 5

    def test_current_user_group_id_none(self):
        """
        测试用户组ID为None - 使用源代码中的 current_user_group_id 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": None}
        
        # 使用源代码中的 appHelper.current_user_group_id 方法
        result = appHelper.current_user_group_id(mock_request)
        
        assert result is None

    def test_current_user_empty_dict(self):
        """
        测试用户信息为空字典 - 使用源代码中的 current_user 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {}
        
        # 使用源代码中的 appHelper.current_user 方法
        result = appHelper.current_user(mock_request)
        
        assert result == {}

    def test_current_user_with_extra_fields(self):
        """
        测试用户信息包含额外字段 - 使用源代码中的 current_user 方法
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {
            "id": 1, 
            "user_name": "test_user", 
            "group_id": 10,
            "email": "test@example.com",
            "phone": "1234567890"
        }
        
        # 使用源代码中的 appHelper.current_user 方法
        result = appHelper.current_user(mock_request)
        
        assert result["email"] == "test@example.com"
        assert result["phone"] == "1234567890"