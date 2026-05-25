from typing import Any, Dict, Generator,TypeVar, Iterator, Tuple, List
# from types import SimpleNamespace # 不支持 data['name'] 及 data.get("name") 这类访问方式

T = TypeVar("T")



class DictNamespace:
    """同时支持属性访问和字典操作的命名空间"""

    def __init__(self, **kwargs):
        """初始化，将所有关键字参数设为属性"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # ========== 字典基础方法 ==========
    def __getitem__(self, key: str) -> Any:
        """支持下标访问: obj['key']"""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """支持下标赋值: obj['key'] = value"""
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        """支持下标删除: del obj['key']"""
        try:
            delattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符: 'key' in obj"""
        return hasattr(self, key)

    def __len__(self) -> int:
        """返回属性数量"""
        return len(self.keys())

    def __iter__(self) -> Iterator[str]:
        """支持迭代: for key in obj"""
        return iter(self.keys())

    # ========== 字典方法 ==========
    def get(self, key: str, default: Any = None) -> Any:
        """获取值，不存在则返回默认值"""
        return getattr(self, key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        """弹出键值对"""
        if not hasattr(self, key):
            if default is not None:
                return default
            raise KeyError(key)
        value = getattr(self, key)
        delattr(self, key)
        return value

    def popitem(self) -> Tuple[str, Any]:
        """弹出最后一个键值对"""
        if not self.__dict__:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(list(self.__dict__.keys())))
        value = self.__dict__[key]
        delattr(self, key)
        return key, value

    def setdefault(self, key: str, default: Any = None) -> Any:
        """设置默认值，如果键不存在则设置并返回默认值"""
        if not hasattr(self, key):
            setattr(self, key, default)
            return default
        return getattr(self, key)

    def update(self, other: Dict = None, **kwargs) -> None:
        """更新字典，接受字典或关键字参数"""
        if other is not None:
            if hasattr(other, 'items'):
                for key, value in other.items():
                    setattr(self, key, value)
            else:
                for key, value in other:
                    setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def clear(self) -> None:
        """清空所有属性"""
        for key in list(self.__dict__.keys()):
            delattr(self, key)

    def copy(self) -> 'DictNamespace':
        """浅拷贝"""
        new_obj = DictNamespace()
        new_obj.__dict__.update(self.__dict__)
        return new_obj

    def keys(self) -> List[str]:
        """返回所有键的列表（过滤掉私有属性）"""
        return [key for key in self.__dict__.keys() if not key.startswith('_')]

    def values(self) -> List[Any]:
        """返回所有值的列表"""
        return [getattr(self, key) for key in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        """返回所有键值对的列表"""
        return [(key, getattr(self, key)) for key in self.keys()]

    # ========== 其他方法 ==========
    def has_key(self, key: str) -> bool:
        """兼容旧式字典方法"""
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        return {key: getattr(self, key) for key in self.keys()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DictNamespace':
        """从字典创建实例"""
        return cls(**data)

    def __repr__(self) -> str:
        """字符串表示"""
        items = ', '.join(f'{k}={v!r}' for k, v in self.items())
        return f"DictNamespace({items})"

    def __eq__(self, other: object) -> bool:
        """比较两个对象是否相等"""
        if not isinstance(other, DictNamespace):
            return False
        return self.to_dict() == other.to_dict()

def fibonacci():
    """无限斐波那契数列生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
def primes_fast():
    """
    素数生成器 
    速度更快 
    埃拉托色尼筛法思想 + 生成器
    """
    composites = {}
    n = 2
    while True:
        if n not in composites:
            yield n
            composites[n * n] = [n]
        else:
            for p in composites[n]:
                composites.setdefault(p + n, []).append(p)
            del composites[n]
        n += 1

def take(gen: Generator[T, None, None], n: int) -> Generator[T, None, None]:
    """从生成器中取前 n 个"""
    for _ in range(n):
        yield next(gen)