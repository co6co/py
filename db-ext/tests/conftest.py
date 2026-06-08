from co6co_db_ext.db_session import db_service, connectSetting
import pytest
import os
from co6co.data import DictNamespace
from co6co_db_ext.actuator import Actuator
from .right import *
import logging

@pytest.fixture
async def db_service_param():
    print("db_service_param...................")
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
    await service.init_tables()
    #service.sync_init_tables()
    actuator = Actuator(service.Session())
    yield cfg, service.Session, actuator
    # 清理：关闭 session
    await actuator.session.close()


@pytest.fixture
async def db_service_param2():
    print("db_service_param...................")
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
    await service.init_tables()
    #service.sync_init_tables().
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    return service