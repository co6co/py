#--*-- coding: utf-8 --*
from __future__ import annotations
from typing import Optional
from co6co.utils import log
from co6co.data import DictNamespace 
import json
from dataclasses import dataclass 
from typing import TypedDict

@dataclass(init=False)
class webConfig:
    port: int
    host: str
    backlog: int
    debug: bool
    access_log: bool
    dev: bool 

    def post_init(self, config: DictNamespace|dict):
        if isinstance(config, dict):
            config = DictNamespace(**config)
        self.port = config.port
        self.host = config.host
        self.backlog = config.backlog
        self.debug = config.debug
        self.access_log = config.access_log
        self.dev = config.dev 

class connectSetting(TypedDict):
    DB_HOST: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    echo: bool
    pool_size: int
    max_overflow: int
    pool_pre_ping: bool
    # 不支持示例方法
    #def from_(self, data: DictNamespace):
    #    for k in self.keys():
    #        self[k] = data.get(k)
    @staticmethod
    def from_(instance:connectSetting, data: DictNamespace):
        for k in instance.keys():
            instance[k] = data.get(k)
    @classmethod
    def create_default(cls, data: DictNamespace = None):
        instance = cls(
            DB_HOST="localhost",
            DB_NAME="",
            DB_USER="root",
            DB_PASSWORD="",
            echo=True,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # 执行sql语句前悲观地检查db是否可用
            # 'pool_recycle':1800 #超时时间 单位s
        )
        if data is not None:
            connectSetting.from_(instance, data)
        return instance
    
    @staticmethod
    def create_from_config(config_file: str,  database_key:str="db_settings"):
        raw:dict={}
        try:
            with open(config_file, "r", encoding="utf-8") as f: 
                raw =json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"配置文件不存在: {config_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"配置文件 JSON 解析失败: {e}")
        db_settings = raw.get(database_key, {})
        print(db_settings)
        if not db_settings:
            raise RuntimeError(f"配置文件中没有 {database_key} 配置")
        instance=connectSetting.create_default()
        connectSetting.from_(instance, DictNamespace(**db_settings))
        return instance


@dataclass
class AppConfig:
    raw: dict
    web: Optional[webConfig] = None
    db: Optional[connectSetting] = None

    @staticmethod
    def get_config(config_file: str|dict, *,use_web_config:bool=True,use_db_config:bool=True):
        appConfig=AppConfig({})
        try:
            if isinstance(config_file, dict):
                appConfig.raw = config_file
            else:
                with open(config_file, "r", encoding="utf-8") as f: 
                    appConfig.raw =json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"配置文件不存在: {config_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"配置文件 JSON 解析失败: {e}")  
        if use_web_config:
            appConfig.web = AppConfig.get_web_config(appConfig.raw)
        if use_db_config: 
            appConfig.db = AppConfig.get_db_config(appConfig.raw)
        return appConfig
    
    @staticmethod
    def get_web_config(configs:dict):
        try: 
            setting = configs.get("web_setting", {})
            web_config = webConfig()
            web_config.post_init(DictNamespace(**setting)) 
            return web_config
        except Exception as e:
            log.err(f"web_setting error:{setting}",e)
            raise e
    @staticmethod
    def get_db_config(configs:dict):
        try:
            setting = configs.get("db_settings", {})
            data=DictNamespace(**setting) 
            _config = connectSetting.create_default(data) 
            return _config
        except Exception as e:
            log.err(f"db_settings error:{setting}",e)
            raise e
 