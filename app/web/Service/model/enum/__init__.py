from __future__ import annotations
from co6co.enums import Base_Enum, Base_EC_Enum
from co6co.utils import DATA


class User_category(Base_Enum):
    """
    用户类别
    """
    sys = "sys", 0  # 系统，需要用户名和密码
    unbind = "unbind", 1  # 未绑定用户信息，即没有用户和密码等信息


class Account_category(Base_Enum):
    """
    账号类别
    """
    wx = "wx", 1  # 微信账号


class CommandCategory(Base_Enum):
    """
    操作类别
    """
    GET = "get", 0  # 获取数据
    Exist = "exist", 1  # 任务是否存在
    START = "start", 2  # 启动任务
    STOP = "stop", 3  # 停止任务
    REMOVE = "remove", 4  # 移除任务
    DELETE = "DELETE", 4  # 移除任务
    RESTART = "restart", 5  # 重启任务
    MODIFY = "modify", 6  # 修改任务

    @staticmethod
    def createOption(command: CommandCategory, data: str = "", success: bool = True, **kwarg) -> DATA:
        return DATA(command=command, data=data, success=success, **kwarg)
