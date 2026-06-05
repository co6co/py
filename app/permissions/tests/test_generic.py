from abc import ABC
from enum import Flag
from typing import Generic, TypeVar
from co6co_db_ext.db_filter import absFilterItems
from typing import Type
from unittest.mock import MagicMock
import pytest
from sqlalchemy import true

FilterType = TypeVar("FilterType", bound=absFilterItems)


class ConcreteFilter(absFilterItems):
    pass


class baseView:
    def __init__(self, request, *args, **kwargs) -> None:
        pass

    def __init_subclass__(cls, init: bool = None, *args, **kwargs) -> None:
        print(f"{cls.__name__}-------------------", init)
        if init:
            super().__init_subclass__(
                *args, **kwargs
            )  # 不调用会阻断 Generic 的设置流程
        pass

@pytest.fixture
def clazz(request ): 
    class AbsView(baseView, Generic[FilterType], init=request.param):
        routePath = "/query"
        pass 
    class testQueryView(AbsView[ConcreteFilter]):
        cls: Type[ConcreteFilter]
        pass 
    return  request.param,testQueryView 


@pytest.mark.parametrize("clazz",[True], indirect=True)   
def test_generic_view(clazz): 
    init,viewClass = clazz
    print("init",init)
    if init:
        view = viewClass(MagicMock())
        assert view is not None 

def test_error():
    class AbsView(baseView, Generic[FilterType], init=False):
        routePath = "/query"
        pass
    with pytest.raises(AttributeError) as exc_info:
        class testQueryView(AbsView[ConcreteFilter]):
            cls: Type[ConcreteFilter]
            pass
        view = testQueryView(MagicMock())  
        print(view)
    print("err->",str(exc_info.value))
    assert "__parameters__" in str(exc_info.value)
def test_subClass():
    class PluginBase:
        def __init_subclass__(cls, init=True, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.init = init
            # print(f"Registering plugin: {cls.__name__}->{init}")

    class MyPlugin(PluginBase):
        pass

    class MyPlugin2(PluginBase, init=False):
        pass

    class MyPluginSub(PluginBase, init=False):
        pass

    plugin = MyPlugin()
    assert plugin is not None
    assert plugin.init is True
    plugin2 = MyPlugin2()
    assert plugin2 is not None
    assert plugin2.init is False
    plugin3 = MyPluginSub()
    assert plugin3 is not None
    assert plugin3.init is False


def test_s2():
    class AbsView2(baseView, Generic[FilterType], init=True):
        routePath = "/query"
        pass

    class testQueryView2(AbsView2[ConcreteFilter]):
        cls: Type[ConcreteFilter]
        pass

    # baseView.__init_subclass__ 会执行两次
    # 1. AbsView2 init=True
    # 2  testQueryView2 init=False
    # 只要有一次为 True baseView 将能执行super().__init_subclass__(*args,**kwargs)
    # 将不会出错

    # AbsView2 中间内传了之后 testQueryView2将无法被传递
    t = testQueryView2(MagicMock())
    print(t)
