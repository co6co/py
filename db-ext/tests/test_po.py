import pytest
from datetime import datetime
from sqlalchemy import Column, String, BigInteger, Integer
from co6co_db_ext.po import (
    BasePO,
    TimeStampedModelPO,
    UserTimeStampedModelPO,
    CreateUserStampedModelPO,
)


class SimpleTestPO(BasePO):
    __tablename__ = "test_simple_po"
    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    name = Column("name", String(64))
    value = Column("value", Integer)


class TestBasePO:
    """测试 BasePO 实体基类"""

    def test_to_dict_basic(self):
        """测试 to_dict 方法基本功能"""
        po = SimpleTestPO()
        po.id = 1
        po.name = "test"
        po.value = 100

        result = po.to_dict()

        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "value" in result
        assert result["id"] == 1
        assert result["name"] == "test"
        assert result["value"] == 100

    def test_to_dict_excludes_sa_instance_state(self):
        """测试 to_dict 排除 SQLAlchemy 实例状态"""
        po = SimpleTestPO()
        po.id = 1
        po.name = "test"

        result = po.to_dict()

        assert "_sa_instance_state" not in result

    def test_sqlItem_basic(self):
        """测试 sqlItem 属性基本功能"""
        po = SimpleTestPO()
        po.id = 1
        po.name = "test"
        po.value = 100

        result = po.sqlItem

        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["name"] == "test"
        assert result["value"] == 100

    def test_sqlItem_excludes_private_attributes(self):
        """测试 sqlItem 排除私有属性"""
        po = SimpleTestPO()
        po.id = 1
        po.name = "test"
        po._private_attr = "should not include"

        result = po.sqlItem

        assert "_private_attr" not in result
        assert "_sa_instance_state" not in result


class TestTimeStampedModelPO:
    """测试 TimeStampedModelPO 时间戳模型"""

    def test_has_createTime_and_updateTime(self):
        """测试包含创建和更新时间戳"""
        from sqlalchemy import Column, String, BigInteger

        class TimestampedTestPO(TimeStampedModelPO):
            __tablename__ = "test_timestamped"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = TimestampedTestPO()
        po.name = "test"

        assert hasattr(po, "createTime")
        assert hasattr(po, "updateTime")


class TestUserTimeStampedModelPO:
    """测试 UserTimeStampedModelPO 用户时间戳模型"""

    def test_has_user_fields(self):
        """测试包含用户相关字段"""
        from sqlalchemy import Column, String, BigInteger

        class UserTimestampedTestPO(UserTimeStampedModelPO):
            __tablename__ = "test_user_timestamped"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = UserTimestampedTestPO()
        po.name = "test"

        assert hasattr(po, "createTime")
        assert hasattr(po, "updateTime")
        assert hasattr(po, "createUser")
        assert hasattr(po, "updateUser")


class TestCreateUserStampedModelPO:
    """测试 CreateUserStampedModelPO 创建用户模型"""

    def test_has_createUser_and_createTime(self):
        """测试包含创建用户和时间戳"""
        from sqlalchemy import Column, String, BigInteger

        class CreateUserTestPO(CreateUserStampedModelPO):
            __tablename__ = "test_create_user"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = CreateUserTestPO()
        po.name = "test"

        assert hasattr(po, "createUser")
        assert hasattr(po, "createTime")


class TestPoAddAssignment:
    """测试 PO 的赋值方法"""

    def test_add_assignment_with_user_timestamped_po(self):
        """测试 UserTimeStampedModelPO 的 add_assignment"""
        from sqlalchemy import Column, String, BigInteger

        class UserTimestampedTestPO(UserTimeStampedModelPO):
            __tablename__ = "test_add_assignment"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = UserTimestampedTestPO()
        po.add_assignment(userId=100)

        assert po.createUser == 100
        assert po.createTime is not None
        assert isinstance(po.createTime, datetime)

    def test_edit_assignment_with_user_timestamped_po(self):
        """测试 UserTimeStampedModelPO 的 edit_assignment"""
        from sqlalchemy import Column, String, BigInteger

        class UserTimestampedTestPO(UserTimeStampedModelPO):
            __tablename__ = "test_edit_assignment"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = UserTimestampedTestPO()
        po.edit_assignment(userId=200)

        assert po.updateUser == 200
        assert po.updateTime is not None
        assert isinstance(po.updateTime, datetime)

    def test_add_assignment_with_timestamped_po(self):
        """测试 TimeStampedModelPO 的 add_assignment"""
        from sqlalchemy import Column, String, BigInteger

        class TimestampedTestPO(TimeStampedModelPO):
            __tablename__ = "test_ts_add_assignment"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = TimestampedTestPO()
        po.add_assignment()

        assert po.createTime is not None
        assert isinstance(po.createTime, datetime)

    def test_edit_assignment_with_timestamped_po(self):
        """测试 TimeStampedModelPO 的 edit_assignment"""
        from sqlalchemy import Column, String, BigInteger

        class TimestampedTestPO(TimeStampedModelPO):
            __tablename__ = "test_ts_edit_assignment"
            id = Column("id", BigInteger, primary_key=True)
            name = Column("name", String(64))

        po = TimestampedTestPO()
        po.edit_assignment()

        assert po.updateTime is not None
        assert isinstance(po.updateTime, datetime)