import pytest
import sys
import os
from unittest.mock import MagicMock 
from co6co_db_ext.appconfig import AppConfig
from sqlalchemy.ext.asyncio import AsyncSession
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="class")
def app_config():
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    # 拼接 config.json 的完整路径
    config_path = os.path.join(current_dir, "test_config.json") 
    config=AppConfig.get_config(config_path)
    return config


@pytest.fixture(scope="class")
def mock_request(app_config:AppConfig): 
    mock_request=MagicMock()
    mock_request.app.config=app_config.raw
    mock_request.ctx.session=MagicMock(spec=AsyncSession)
    return mock_request
