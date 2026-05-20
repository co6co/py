from __future__ import annotations
from co6co.enums import  Base_EC_Enum 

class dict_state(Base_EC_Enum):
    """
    字典和字典类型使用的状态
    """
    enabled = "enabled", "启用", 1
    disabled = "disabled", "禁用", 0