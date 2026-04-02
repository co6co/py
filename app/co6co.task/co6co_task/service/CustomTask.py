
from __future__ import annotations
from abc import ABC, abstractmethod
# 定义接口
from typing import overload, List, Tuple
from co6co.utils import log
from typing import TypeVar

from model.enum import Category
T = TypeVar('T', bound='ICustomTask')


class ICustomTask(ABC):
    """
    需要任务类实现该类，才能被识别为任务
    """
    name = "抽象任务接口"
    code = "ICustomTask"

    def __init__(self):
        super().__init__()
        self._isQuit = False
        self.name = self.__class__.name
        self.code = self.__class__.code

    @property
    def isQuit(self) -> bool:
        return self._isQuit

    @abstractmethod
    def main(self):
        pass

    def stop(self):
        self._isQuit = True
        pass

    @staticmethod
    def createInstance(cls: T, code: str) -> T:
        """获取code子类的实例"
        """
        class_arr = get_all_subclasses(cls)
        for c in class_arr:
            if c.code == code:
                try:
                    return c()
                except Exception as e:
                    log.err(f"实例化'{c}'失败", e)
        return None


class ICustomService(ABC):
    name = "抽象服务"
    code = "ICustomService"

    def __init__(self):
        super().__init__()
        self._isQuit = False
        self._running = False
        self.name = self.__class__.name
        self.code = self.__class__.code

    @property
    def running(self) -> bool:
        return self._running

    @property
    def isQuit(self) -> bool:
        """
        抽象类方法不改变 _running 的状态，由子类实现
        """
        return self._isQuit

    @abstractmethod
    def main(self):
        pass

    def start(self):
        if self._running:
            log.warn(f"{self.name}已经在运行中")
            return
        self.main()

    def stop(self):
        self._isQuit = True
        pass

    @staticmethod
    def createInstance(cls, code: str):
        """获取code子类的实例"
        """
        class_arr = get_all_subclasses(cls)
        for c in class_arr:
            if c.code == code:
                try:
                    return c()
                except Exception as e:
                    log.err(f"实例化'{c}'失败", e)
        return None


def get_all_subclasses(cls=ICustomTask):
    """
    获取所有子类的列表
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def get_list() -> List[Tuple[str, str]]:
    """
    获取所有子类的列表
    [(name,code),]
    """
    class_arr = get_all_subclasses()
    _list = [(c.name, c.code)for c in class_arr]
    return _list
