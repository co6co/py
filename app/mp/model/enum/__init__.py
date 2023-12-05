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
class resource_image_sub_category(Base_Enum):
    """
    图片子资源类型
    """
    raw="raw",0
    marked="marked",1

class wx_menu_state(Base_EC_Enum):
    unpushed="unpushed","未推送",0
    pushed="pushed","已推送",1
    failed="failed","推送失败",9

class device_type(Base_Enum):
     box="盒子",1
     ip_camera="网络摄像机",2