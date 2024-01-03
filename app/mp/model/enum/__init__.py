from co6co.enums import Base_Enum,Base_EC_Enum

class User_Group(Base_Enum):
    """
    用户组代码
    """
    wx_user="wx_user",0
    wx_admin="wx_admin",1

class User_Role(Base_Enum):
    """
    角色代码
    """
    wx_user="wx_user_role",0
    wx_admin="wx_admin_role",1

class User_category(Base_Enum):
    """
    用户类别
    """
    sys="sys",0 # 系统，需要用户名和密码
    unbind="unbind",1 # 未绑定用户信息，即没有用户和密码等信息
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
    image="image",1
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
     """
     设备类型
     """
     box="盒子",1
     ip_camera="网络摄像机",2
     router="路由器",3
     mqqt_server="mqtt服务器",4
     xss_server="语音对讲服务器",5
     sip_server="sip服务器",6