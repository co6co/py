from __future__ import annotations
from co6co.enums import  Base_EC_Enum 

class resource_category(Base_Enum):
    """
    资源类型
    """
    image = "image", 0
    video = "video", 1
    file = "file", 2

 

class menu_type(Base_EC_Enum):
    """
    菜单类型 
    """
    group = "group", "分组", 0  # 分组菜单
    api = "api", "API接口", 1   # api
    view = "view", "页面视图", 2  # a视图
    subView = "subView", "页面子视图", 3  # a视图
    button = "button", "视图功能", 10  # 视图中的按钮等。


class menu_state(Base_EC_Enum):
    """
    菜单类型 
    """

    enabled = "enabled", "启用", 0
    disabled = "disabled", "禁用", 1