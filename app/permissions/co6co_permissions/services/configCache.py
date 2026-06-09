
from sanic import Request
from sqlalchemy.sql import Select
from co6co.utils import log
from co6co_db_ext.actuator import Actuator
from .baseCache import CustomSanicCache
from co6co_db_ext.db_utils import DbCallable
from ..model.pos.other import sysConfigPO
import json as sysJson
from typing import   TypedDict


async def get_config(request: Request, code: str, *, useDist=False, default: any = None) -> str | dict | None:
    """
    获取配置
    """
    try:
        cache = ConfigCache(request)
        result = cache.getConfig(code)
        if result is None:
            await cache.queryConfig(code)
            result = cache.getConfig(code)
        if result and useDist:
            result = sysJson.loads(result) if isinstance(result, str) else result
            # 　如果default是dict,则合并，
            merge_dict = {}
            if default and isinstance(default, dict):
                merge_dict.update(default)
            merge_dict.update(result)
            result = merge_dict
        if result is None:
            log.warn("未找到'{}'的相关配置,使用默认配置：{}".format(code, default))
            result = default
        return result
    except Exception as e:
        log.err("获取配置失败:{},使用默认配置：{}".format(e, default), e)
        return default


async def get_upload_path(request: Request) -> str:
    """
    获取上传路径
    数据库未配置使用 /upload
    """
    key = "SYS_CONFIG_UPLOAD_PATH"
    return await get_config(request, key, default="/upload")


async def get_terminal_access_token(request: Request) -> str:
    """
    获取终端配置
    """
    cache = ConfigCache(request)
    cache.setCache("SYS_CONFIG_TERMINAL_ACCESS_TOKEN", "123456")
    # 获取
    return cache.getConfig("SYS_CONFIG_TERMINAL_ACCESS_TOKEN")


class UserConfig(TypedDict):
    loginFail: int
    lockSeconds: int


async def get_user_config(request: Request) -> UserConfig:
    key = "SYS_USER_CONFIG"
    default = {"loginFail": 3, "lockSeconds": 5*60}
    config = await get_config(request, key, useDist=True, default=default)

    return config


class ConfigCache(CustomSanicCache):

    def __init__(self, request: Request) -> None:
        super().__init__(request)

    @property
    def configKeyPrefix(self):
        return 'ConfigKey'

    def getKey(self, code: str):
        """
        获取Key
        """
        return "{}_{}".format(self.configKeyPrefix, code)

    def _convertValue(self, value: str) -> str | dict:
        """
        转换配置值
        """
        if "{" in value and "}" in value:
            try:
                value = sysJson.loads(value)
            except Exception as e:
                log.err("load json config {} error:{}".format(value, e))
                raise Exception("load json config error,check value  :{}".format(value)) 
        return value
    async def queryConfig(self, code: str) -> str | None:
        """
        查询当前用户的所拥有的角色
        结果放置在cache中
        """
        callable = DbCallable(self.dbService.Session())

        async def exe(actuator:Actuator) -> str | None:
            select = (
                Select(sysConfigPO.name, sysConfigPO.code,
                       sysConfigPO.value, sysConfigPO.remark)
                .filter(sysConfigPO.code.__eq__(code))
            )
            data  = await actuator.query_one_mappings(select)
            
            result = None
            if data is None:
                log.warn("query {} config is NULL".format(code))
            else:
                result = data.get("value")
                # 配置的是否时json字符串
                result=self._convertValue(result)
                self.setConfig(code, result)
            return result

        return await callable(exe)

    def setConfig(self, code: str, value: str):
        if code is not None:
            result=self._convertValue(value)
            self.setCache(self.getKey(code), result)

    def getConfig(self, code: str) -> str | None:
        if code is not None:
            return self.getCache(self.getKey(code))

    def removeConfig(self, code: str) -> str | None:
        if code is not None:
            return self.remove(self.getKey(code))
