
from co6co.enums import Base_Enum, Base_EC_Enum
import pytest
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Color2(Base_Enum):
    RED = 'red', 1
    GREEN = 'green', 2
    BLUE = 'blue', 3


class DemoEnum(Base_EC_Enum):
    A = "a", "LA", 88
    B = ("b", "LB", 99)
    C = "c", "LC", 100


def test_to_str():
    print("颜色", Color2.to_str("key", "name"))
    print(Color2.to_str("name", 'key'))

    print("颜色", DemoEnum.to_str("key", "name"))
    print(DemoEnum.to_str("label", 'name'))
    print(DemoEnum.to_labels_str())


def test_rawenum():
    color = Color(1)
    print(color.name, color.value)
    assert color == Color.RED


def test_enum2():
    # demo = DemoEnum("A") # 不支持该方法
    pass


def test_enum():
    enum = DemoEnum.key2enum("a")
    assert enum == DemoEnum.A
    enum = DemoEnum.val2enum(99)
    assert enum == DemoEnum.B
    assert DemoEnum.A.value == 1
    print(DemoEnum.to_dict_list())
    print(DemoEnum.to_labels_str())
    print("DemoEnum.value_of", DemoEnum.value_of("a", True))
    assert DemoEnum.value_of("a", True) == None
    assert DemoEnum.value_of("A") == DemoEnum.A

    gen = DemoEnum.generator()
    a = next(gen)
    assert a.value == 1
    assert a.val == 88
    assert a.key == "a"
    assert a.label == "LA"
    assert a.name == 'A'
    for i in gen:
        print(i, id(i), id(DemoEnum.B))
