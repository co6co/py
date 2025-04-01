
from .base import ICustomTask
from .cfTask import CfTaskMgr
from typing import overload, List, Tuple
from co6co.utils import log
__all__ = ['ICustomTask', "CfTaskMgr"]


@overload
def get_all_subclasses():
    """
    获取所有子类的列表
    """
    pass


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


def get_task(code: str) -> ICustomTask | None:
    """
    获取所有子类的列表"
    """
    class_arr = get_all_subclasses()
    for c in class_arr:
        if c.code == code:
            try:
                return c()
            except Exception as e:
                log.err(f"实例化'{c}'失败", e)
    return None
