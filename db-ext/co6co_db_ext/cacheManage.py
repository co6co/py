from multiprocessing.managers import DictProxy
from co6co_db_ext.db_session import db_service
from sqlalchemy.ext.asyncio import AsyncSession


class CacheManage:
    @property
    def dbSession(self):
        """
        创建Session
        请自行管理
        """
        return self._session

    @property
    def DbSessionFactory(self):
        """
        创建Session
        请自行管理
        """
        return self.dbService.Session

    def __init__(
        self,
        cache: DictProxy,
        *,
        session: AsyncSession = None,
        db_service: db_service = None,
    ) -> None:
        self._cache=cache
        self.dbService = db_service
        self._session: AsyncSession = session
        pass

    @property
    def cache(self) -> DictProxy:
        """
        缓存
        """
        return self._cache

    def setCache(self, key: str, value: any):
        """
        设置数据缓存
        """
        self.cache[key] = value

    def getCache(self, key: str):
        """
        获取数据缓存
        """
        if key in self.cache:
            return self.cache[key]
        return None

    def get(self, key: str, default: any = None):
        """
        获取数据缓存
        """
        return self.cache.get(key, default)

    def exist(self, key: str):
        """
        是否存在
        """
        return key in self.cache

    def remove(self, key: str):
        """
        移除缓存 key
        return key对应的值,没有返回空
        """
        # del my_dict['b']  key 可以必须存在， KeyError 异常
        return self.cache.pop(key, None)
