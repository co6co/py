from co6co.enums import Base_Enum,Base_EC_Enum

class Account_category(Base_Enum):
    """
    账号类别
    """
    wx="wx",1 # 微信账号
class resource_category(Base_Enum):
    """
    资源类型
    """
    video="video",0
    image="image",0

class wx_menu_state(Base_EC_Enum):
    unpushed="unpushed","未推送",0
    pushed="pushed","已推送",1
    failed="failed","推送失败",9