from co6co_db_ext.db_session import db_service, connectSetting
import pytest
import os
from co6co.data import DictNamespace
from co6co_db_ext.actuator import Actuator
from .right import *
 

@pytest.fixture
def db_service_param():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接 config.json 的完整路径
    config_path = os.path.join(current_dir, "test_config.json")
    cfg = connectSetting.create_from_config(config_path)
    cfg = DictNamespace(**cfg)
    assert cfg is not None
    assert cfg.DB_HOST is not None
    assert cfg.DB_NAME is not None
    assert cfg.DB_USER is not None
    assert cfg.DB_PASSWORD is not None
    service = db_service(cfg)
    service.sync_init_tables()
    return cfg, service.Session,   Actuator(service.Session())
