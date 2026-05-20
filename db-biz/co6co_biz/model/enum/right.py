
from __future__ import annotations
from co6co.enums import  Base_EC_Enum 

class user_category(Base_EC_Enum):
    normal = "normal", "普通", 0
    system = "system", "系统", 1
    terminal = "terminal ", "终端", 2

class account_category(Base_EC_Enum):
    wx = "wx", "微信", 100 

class user_state(Base_EC_Enum):
    enabled = "enabled", "启用", 0
    disabled = "disabled", "禁用", 1
    locked = "locked", "锁定", 2
