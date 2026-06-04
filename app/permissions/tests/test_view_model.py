"""
测试 co6co_permissions.view_model 模块中的视图类
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from co6co_permissions.view_model.base_view import AbsClsView, _authView
from co6co_permissions.model.pos.right import UserPO


class TestAbsClsView:
    """
    测试基础类视图 AbsClsView
    使用源代码中的 AbsClsView 类进行测试
    """

    def test_create_token_with_none_user(self):
        """
        测试用户为None时create_token返回None - 使用源代码中的方法
        """
        mock_request = MagicMock()
        mock_request.headers = {"User-Agent": "test_agent"}
        
        mock_app_config = MagicMock()
        mock_app_config.raw = {"SECRET": "test_secret"}
        
        # 使用源代码中的 AbsClsView 类
        view = AbsClsView(mock_request)
        
        # 测试源代码中的 create_token 方法
        result = view.create_token(None)
        
        assert result is None

    def test_create_token_with_valid_user(self):
        """
        测试有效用户创建token - 使用源代码中的方法
        """
        mock_request = MagicMock()
        mock_request.headers = {"User-Agent": "test_agent"}
        from co6co_db_ext.appconfig import AppConfig
        import os
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接 config.json 的完整路径
        config_path = os.path.join(current_dir, "test_config.json")
        mock_request.app.config=AppConfig.get_config(config_path)

        
        
        # 使用源代码中的 UserPO 类创建用户
        user = UserPO()
        user.id = 1
        user.userName = "test_user"
        user.userGroupId = 10
        user.version = 1
        
        # 使用源代码中的 AbsClsView 类
        view = AbsClsView(mock_request )
        
        # 测试源代码中的 create_token 方法
        result = view.create_token(user)
        
        # token应该是一个字典，包含token和refreshToken
        assert result is not None
        assert isinstance(result, dict)


class TestAuthView:
    """
    测试认证视图 _authView
    使用源代码中的 _authView 类进行测试
    """

    def test_current_user_property(self):
        """
        测试current_user属性 - 使用源代码中的属性
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": 10}
        
        # 使用源代码中的 _authView 类
        view = _authView(mock_request, MagicMock())
        
        # 测试源代码中的 current_user 属性
        result = view.current_user
        
        assert result is not None
        assert result["id"] == 1
        assert result["user_name"] == "test_user"
        assert result["group_id"] == 10

    def test_user_id_property(self):
        """
        测试userId属性 - 使用源代码中的属性
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 123, "user_name": "test_user", "group_id": 10}
        
        # 使用源代码中的 _authView 类
        view = _authView(mock_request, MagicMock())
        
        # 测试源代码中的 userId 属性
        result = view.userId
        
        assert result == 123
        assert isinstance(result, int)

    def test_user_name_property(self):
        """
        测试userName属性 - 使用源代码中的属性
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "admin", "group_id": 10}
        
        # 使用源代码中的 _authView 类
        view = _authView(mock_request, MagicMock())
        
        # 测试源代码中的 userName 属性
        result = view.userName
        
        assert result == "admin"

    def test_group_id_property(self):
        """
        测试groupId属性 - 使用源代码中的属性
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": 5}
        
        # 使用源代码中的 _authView 类
        view = _authView(mock_request, MagicMock())
        
        # 测试源代码中的 groupId 属性
        result = view.groupId
        
        assert result == 5

    def test_group_id_property_none(self):
        """
        测试groupId属性为None - 使用源代码中的属性
        """
        mock_request = MagicMock()
        mock_request.ctx.__dict__["current_user"] = {"id": 1, "user_name": "test_user", "group_id": None}
        
        # 使用源代码中的 _authView 类
        view = _authView(mock_request, MagicMock())
        
        # 测试源代码中的 groupId 属性
        result = view.groupId
        
        assert result is None


class TestDecorators:
    """
    测试装饰器功能
    使用源代码中的装饰器进行测试
    """

    def test_ctx_decorator_import(self):
        """
        测试ctx装饰器可以正常导入
        """
        from co6co_permissions.view_model.aop.api_auth import ctx
        
        # 验证装饰器是一个函数
        assert ctx is not None
        assert callable(ctx)

    def test_authorized_decorator_import(self):
        """
        测试authorized装饰器可以正常导入
        """
        from co6co_permissions.view_model.aop.api_auth import authorized
        
        # 验证装饰器是一个函数
        assert authorized is not None
        assert callable(authorized)


class TestViewClasses:
    """
    测试视图类可以正常导入和使用
    """

    def test_ctx_method_view_import(self):
        """
        测试CtxMethodView可以正常导入
        """
        from co6co_permissions.view_model.base_view import CtxMethodView
        
        # 验证类存在
        assert CtxMethodView is not None

    def test_auth_method_view_import(self):
        """
        测试AuthMethodView可以正常导入
        """
        from co6co_permissions.view_model.base_view import AuthMethodView
        
        # 验证类存在
        assert AuthMethodView is not None

    def test_change_pwd_view_import(self):
        """
        测试changePwd_view可以正常导入
        """
        from co6co_permissions.view_model.currentUser import changePwd_view
        
        # 验证类存在
        assert changePwd_view is not None
        assert hasattr(changePwd_view, 'routePath')
        assert changePwd_view.routePath == "/changePwd"

    def test_user_avatar_view_import(self):
        """
        测试user_avatar_view可以正常导入
        """
        from co6co_permissions.view_model.currentUser import user_avatar_view
        
        # 验证类存在
        assert user_avatar_view is not None
        assert hasattr(user_avatar_view, 'routePath')
        assert user_avatar_view.routePath == "/avatar"

    def test_user_info_view_import(self):
        """
        测试user_info_view可以正常导入
        """
        from co6co_permissions.view_model.currentUser import user_info_view
        
        # 验证类存在
        assert user_info_view is not None
        assert hasattr(user_info_view, 'routePath')
        assert user_info_view.routePath == "/currentUser"


class TestAuthServiceImport:
    """
    测试AuthService可以正常导入和使用
    """

    def test_auth_service_import(self):
        """
        测试AuthService可以正常导入
        """
        from co6co_permissions.services.authService import AuthService
        
        # 验证类存在
        assert AuthService is not None

    def test_auth_service_properties(self):
        """
        测试AuthService的属性
        """
        from co6co_permissions.services.authService import AuthService
        
        mock_request = MagicMock()
        mock_request.app.config = MagicMock()
        mock_request.app.config.raw = {"SECRET": "test_secret"}
        mock_request.token = "test_token"
        
        # 使用源代码中的 AuthService 类
        service = AuthService(mock_request)
        
        # 测试源代码中的属性
        assert service.token == "test_token"
        assert service.jwtService is not None
        assert service.sanicCache is not None