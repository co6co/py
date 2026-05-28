import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from co6co_db_ext.session import session_context, transactional, dbBll
from co6co_db_ext.db_session import connectSetting


class TestSessionContext:
    """测试 session_context 会话上下文管理器"""

    def test_init(self):
        """测试初始化"""
        mock_session = MagicMock()
        context = session_context(mock_session)
        assert context.session == mock_session

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试上下文管理器"""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))

        context = session_context(mock_session)

        async with context() as session:
            pass


class TestTransactionalDecorator:
    """测试 transactional 装饰器"""

    @pytest.mark.asyncio
    async def test_transactional_decorator(self):
        """测试事务装饰器"""
        class MockClass:
            session = MagicMock()

            @transactional
            async def test_method(self, session, arg1):
                return arg1

        mock_instance = MockClass()

        mock_instance.session.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_instance.session.__aexit__ = AsyncMock(return_value=None)

        mock_trans_manager = MagicMock()
        mock_trans_manager.__aenter__ = AsyncMock()
        mock_trans_manager.__aexit__ = AsyncMock(return_value=None)
        mock_instance.session.begin = MagicMock(return_value=mock_trans_manager)

        result = await mock_instance.test_method("test_arg")
        assert result == "test_arg"


class TestDbBll:
    """测试 dbBll 类"""

    def test_init_empty_settings_raises_exception(self):
        """测试空设置参数抛出异常"""
        with pytest.raises(Exception):
            dbBll()

    def test_init_with_settings(self):
        """测试带设置参数初始化"""
        settings = connectSetting.create_default()
        settings["DB_NAME"] = "test_db"

        bll = dbBll(db_settings=settings)
        assert bll.db_settings is not None
        assert bll.closed is False

        bll.close()

    def test_str_representation(self):
        """测试字符串表示"""
        settings = connectSetting.create_default()
        settings["DB_NAME"] = "test_db"

        bll = dbBll(db_settings=settings)
        result = str(bll)
        assert "dbBll" in result
        bll.close()

    def test_close(self):
        """测试关闭"""
        settings = connectSetting.create_default()
        settings["DB_NAME"] = "test_db"

        bll = dbBll(db_settings=settings)
        bll.close()

        assert bll.closed is True
