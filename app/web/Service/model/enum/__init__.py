from __future__ import annotations
from co6co.enums import Base_Enum, Base_EC_Enum
from co6co.utils import DATA
from datetime import datetime


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
    monitor = 'monitor', "监控设备", 0
    # 卡口
    Flow = 'flow', "流量设备", 10
    # 电警
    Warn = 'Warn', "警戒设备", 11
    # 一体机 有两个摄像头
    ParkAndPass = "ParkAndPass", "一体机", 12
    storage = "storage", "录像机", 100,
    server = "server", "服务器", 101,

    @staticmethod
    def getMonitor():
        return [
            DeviceCategory.monitor,
            DeviceCategory.Flow,
            DeviceCategory.Warn,
            DeviceCategory.ParkAndPass,
        ]

    def hasVideo(self):
        """
        是路口设备
        """
        return self in DeviceCategory.getMonitor()

    def isNetworkDev(self):
        """
        是网络设备
        """
        return True


class DeviceCheckState(Base_EC_Enum):
    normal = 'normal', "正常", 0
    networkError = 'networkError', "网络异常", 1
    videoError = 'videoError', "视频异常", 2
    abnormal = 'abnormal', "位置异常", 100


class DeviceVender(Base_EC_Enum):
    Uniview = 'UNIVIEW', "宇视", 1
    Hikvision = 'HIKVISION', "海康威视", 2
    Dahua = 'DAHUA', "大华", 3
    TPLink = 'TPLINK', "天润", 4
    