from co6co.enums import Base_Enum, Base_EC_Enum


class User_Group(Base_Enum):
    """
    用户组代码
    差不数据表时 不能使用 val
    """
    wx_user = "wx_user", 0

    wx_admin = "wx_admin", 1
    wx_alarm = "wx_alarm", 2  # 告警订阅订阅组


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


class wx_menu_state(Base_EC_Enum):
    unpushed = "unpushed", "未推送", 0
    pushed = "pushed", "已推送", 1
    failed = "failed", "推送失败", 9
