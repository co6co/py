"""
测试 co6co_permissions.model 模块中的数据模型
"""
import pytest
from co6co_permissions.model.pos.right import UserPO, RolePO, UserGroupPO, menuPO
from co6co_permissions.model.enum import user_category, user_state, menu_type, dict_state, resource_category


class TestUserPO:
    """
    测试用户模型 UserPO
    """

    def test_generate_salt(self):
        """
        测试生成salt方法
        """
        salt = UserPO.generateSalt()
        assert salt is not None
        assert isinstance(salt, str)
        assert len(salt) == 6

    def test_encrypt_password(self):
        """
        测试密码加密方法
        """
        user = UserPO()
        user.salt = "abc123"
        user.password = "test_password"
        
        # 使用源代码中的 encrypt 方法
        encrypted = user.encrypt()
        assert encrypted is not None
        assert isinstance(encrypted, str)
        assert len(encrypted) == 32  # MD5 结果长度

    def test_encrypt_with_external_password(self):
        """
        测试使用外部密码进行加密
        """
        user = UserPO()
        user.salt = "salt123"
        
        # 使用源代码中的 encrypt 方法，传入外部密码
        encrypted = user.encrypt("external_pwd")
        assert encrypted is not None
        assert isinstance(encrypted, str)

    def test_verify_password_correct(self):
        """
        测试验证正确密码 - 使用源代码中的 verifyPwd 方法
        """
        user = UserPO()
        user.salt = "abc123"
        plain_password = "correct_password"
        # 使用源代码中的 encrypt 方法加密密码
        user.password = user.encrypt(plain_password)
        
        # 使用源代码中的 verifyPwd 方法验证
        result = user.verifyPwd(plain_password)
        assert result is True

    def test_verify_password_incorrect(self):
        """
        测试验证错误密码 - 使用源代码中的 verifyPwd 方法
        """
        user = UserPO()
        user.salt = "abc123"
        plain_password = "correct_password"
        user.password = user.encrypt(plain_password)
        
        # 使用源代码中的 verifyPwd 方法验证错误密码
        result = user.verifyPwd("wrong_password")
        assert result is False

    def test_jwt_data_property(self):
        """
        测试 jwt_data 属性 - 使用源代码中的属性
        """
        user = UserPO()
        user.id = 1
        user.userName = "test_user"
        user.userGroupId = 10
        
        # 使用源代码中的 jwt_data 属性
        jwt_data = user.jwt_data
        assert jwt_data is not None
        assert isinstance(jwt_data, dict)
        assert jwt_data["id"] == 1
        assert jwt_data["user_name"] == "test_user"
        assert jwt_data["group_id"] == 10

    def test_jwt_data_with_none_group_id(self):
        """
        测试 jwt_data 属性 - 用户组ID为None的情况
        """
        user = UserPO()
        user.id = 2
        user.userName = "admin"
        user.userGroupId = None
        
        jwt_data = user.jwt_data
        assert jwt_data["group_id"] is None

    def test_from_jwt_data(self):
        """
        测试从jwt_data恢复用户信息 - 使用源代码中的 from_jwt_data 方法
        """
        user = UserPO()
        jwt_data = {"id": 1, "user_name": "test_user", "group_id": 5}
        
        # 使用源代码中的 from_jwt_data 方法
        user.from_jwt_data(jwt_data)
        assert user.id == 1
        assert user.userName == "test_user"
        assert user.userGroupId == 5

    def test_from_jwt_data_with_none_group_id(self):
        """
        测试从jwt_data恢复用户信息 - group_id为None
        """
        user = UserPO()
        jwt_data = {"id": 1, "user_name": "test_user", "group_id": None}
        
        user.from_jwt_data(jwt_data)
        assert user.userGroupId is None

    def test_create_jwt_refresh_data(self):
        """
        测试创建JWT刷新数据 - 使用源代码中的 crate_jwt_refresh_data 方法
        """
        user = UserPO()
        user.id = 1
        user.version = 2
        
        # 使用源代码中的 crate_jwt_refresh_data 方法
        refresh_data = user.crate_jwt_refresh_data(device="device123", userAgent="test_agent")
        assert refresh_data is not None
        assert isinstance(refresh_data, dict)
        assert refresh_data["id"] == 1
        assert refresh_data["deviceId"] == "device123"
        assert refresh_data["userAgent"] == "test_agent"
        assert refresh_data["version"] == 2

    def test_update_user(self):
        """
        测试更新用户信息 - 使用源代码中的 update 方法
        """
        user = UserPO()
        user.id = 1
        user.userName = "old_name"
        user.category = user_category.normal.val
        user.userGroupId = 1
        user.state = user_state.enabled.val
        user.remark = "old remark"
        user.version = 0
        
        # 创建更新数据
        update_user = UserPO()
        update_user.userName = "new_name"
        update_user.category = user_category.system.val
        update_user.userGroupId = 5
        update_user.state = user_state.disabled.val
        update_user.remark = "new remark"
        
        # 使用源代码中的 update 方法
        user.update(update_user)
        
        assert user.userName == "new_name"
        assert user.category == user_category.system.val
        assert user.userGroupId == 5
        assert user.state == user_state.disabled.val
        assert user.remark == "new remark"
        assert user.version == 1


class TestRolePO:
    """
    测试角色模型 RolePO
    """

    def test_update_role(self):
        """
        测试更新角色信息 - 使用源代码中的 update 方法
        """
        role = RolePO()
        role.id = 1
        role.code = "old_code"
        role.name = "old_name"
        role.remark = "old remark"
        role.order = 1
        
        update_role = RolePO()
        update_role.code = "new_code"
        update_role.name = "new_name"
        update_role.remark = "new remark"
        update_role.order = 10
        
        # 使用源代码中的 update 方法
        role.update(update_role)
        
        assert role.code == "new_code"
        assert role.name == "new_name"
        assert role.remark == "new remark"
        assert role.order == 10


class TestUserGroupPO:
    """
    测试用户组模型 UserGroupPO
    """

    def test_update_user_group(self):
        """
        测试更新用户组信息 - 使用源代码中的 update 方法
        """
        group = UserGroupPO()
        group.id = 1
        group.code = "old_code"
        group.name = "old_name"
        group.parentId = 1
        group.order = 1
        
        update_group = UserGroupPO()
        update_group.code = "new_code"
        update_group.name = "new_name"
        update_group.parentId = 2
        update_group.order = 5
        
        # 使用源代码中的 update 方法
        group.update(update_group)
        
        assert group.code == "new_code"
        assert group.name == "new_name"
        assert group.parentId == 2
        assert group.order == 5


class TestMenuPO:
    """
    测试菜单模型 menuPO
    """

    def test_update_menu(self):
        """
        测试更新菜单信息 - 使用源代码中的 update 方法
        """
        menu = menuPO()
        menu.id = 1
        menu.code = "old_code"
        menu.name = "old_name"
        menu.parentId = 1
        menu.url = "/old/url"
        menu.methods = "GET"
        menu.order = 1
        menu.category = menu_type.api.val
        menu.icon = "old_icon"
        menu.component = "old_component"
        menu.permissionKey = "old_key"
        menu.status = 0
        menu.remark = "old remark"
        
        update_menu = menuPO()
        update_menu.code = "new_code"
        update_menu.name = "new_name"
        update_menu.parentId = 2
        update_menu.url = "/api/test"
        update_menu.methods = "GET,POST"
        update_menu.order = 5
        update_menu.category = menu_type.view.val
        update_menu.icon = "new_icon"
        update_menu.component = "new_component"
        update_menu.permissionKey = "new_key"
        update_menu.status = 1
        update_menu.remark = "new remark"
        
        # 使用源代码中的 update 方法
        menu.update(update_menu)
        
        assert menu.code == "new_code"
        assert menu.name == "new_name"
        assert menu.parentId == 2
        assert menu.url == "/api/test"
        assert menu.methods == "GET,POST"
        assert menu.order == 5
        assert menu.category == menu_type.view.val
        assert menu.icon == "new_icon"
        assert menu.component == "new_component"
        assert menu.permissionKey == "new_key"
        assert menu.status == 1
        assert menu.remark == "new remark"


class TestEnums:
    """
    测试枚举类
    """

    def test_user_category_values(self):
        """
        测试用户类别枚举值
        """
        assert user_category.normal.val == 0
        assert user_category.system.val == 1
        assert user_category.terminal.val == 2

    def test_user_state_values(self):
        """
        测试用户状态枚举值
        """
        assert user_state.enabled.val == 0
        assert user_state.disabled.val == 1
        assert user_state.locked.val == 2

    def test_menu_type_values(self):
        """
        测试菜单类型枚举值
        """
        assert menu_type.group.val == 0
        assert menu_type.api.val == 1
        assert menu_type.view.val == 2
        assert menu_type.subView.val == 3
        assert menu_type.button.val == 10

    def test_dict_state_values(self):
        """
        测试字典状态枚举值
        """
        assert dict_state.enabled.val == 1
        assert dict_state.disabled.val == 0

    def test_resource_category_values(self):
        """
        测试资源类型枚举值
        """
        assert resource_category.image.val == 0
        assert resource_category.video.val == 1
        assert resource_category.file.val == 2