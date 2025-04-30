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


class TaskStatue(Base_Enum):
    RUNNING = "running", 1  # 运行中
    STOPPED = "stopped", 0  # 已停止
    ERROR = "error", 2  # 错误
    UNKNOWN = "unknown", 3  # 未知


class DeviceCategory(Base_EC_Enum):
    # 卡口
    Pass = 'pass', "卡口设备", 0
    # 电警
    Ele = 'ele', "电警设备", 1

    # 违停身边
    Park = 'park', "违停设备", 2
    # 一体机
    ParkAndPass = 'parkAndPass', "一体机设备", 3
    monitor = 'monitor', "监控", 4
    # 一体机
    other = 'other', "其他设备", 9

    