from co6co_db_ext.po import BasePO ,metadata
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import  relationship,declarative_base
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL 
from sqlalchemy import MetaData  
import uuid

#metadata = MetaData() 
#BasePO = declarative_base(metadata=metadata)
class UserPO(BasePO):
    __tablename__ = "sys_user" 
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    userName = Column("user_name",String(64), unique=True,  comment="userName")
    category= Column("category",Integer,)
    password = Column("user_pwd",String(256),comment="密码")
    salt=Column("user_salt",String(64),comment="pwd=盐")
    userGroup=Column("user_group_id",ForeignKey(f"sys_user_group.id"))
    state=Column("state",INTEGER, comment="用户状态:0启用,1锁定,2禁用",server_default='0', default=0)  
    avatar=Column("avatar",String(256),comment="图像URL")
    remark=Column("remark",String(1500),comment="备注")
    createTime=Column("create_time",DateTime , server_default=func.now() )
    create_user=Column("create_user",BigInteger)
    @staticmethod
    def generateSalt( )->str:
        """
        生成salt
        """
        return tool.getRandomStr(6)
    def encrypt(self,password:str=None)->str:
        """
        加密密码
        不保存到属性
        """ 
        if password!=None: return hash.md5(self.salt+password) 
        return hash.md5(self.salt+self.password) 
    def verifyPwd(self,plainText:str)->bool:
        """
        验证输入的密码是否正确
        """
        return hash.md5(self.salt+plainText)==self.password
    def to_dict(self):
        """
        jwt 保存内容
        """
        return {"id":self.id,"userName":self.userName, "group_id":self.userGroup}

class AccountPO(BasePO):
    """
    账号：以各种方式登录系统 用户名  EMAIL openID
    """
    __tablename__ = "sys_account" 
    uid = Column("uuid",String(36),  primary_key=True,default=uuid.uuid1())
    #id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    user=Column("user_id",ForeignKey(f"{UserPO.__tablename__}.{UserPO.id.key}"))
    accountName = Column("account_name",String(64), unique=True)
    attachInfo= Column("attach_info",String(255),comment="有些信息需要存下来，才能配合账号使用")
    category = Column("category",INTEGER)
    status = Column("status",String(16),comment="状态,根据账号类型，有不同得解释")
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")
    createUser= Column("create_user", BigInteger,comment="创建人")                   
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人") 

class UserGroupPO(BasePO):
    """
    用户组现在还不知道用来做什么
    """
    __tablename__ = "sys_user_group" 
    id = Column("id",Integer,autoincrement=True, primary_key=True)
    name = Column("group_name",String(64), unique=True) 
    createUser= Column("create_user", BigInteger,comment="创建人")  
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")                 
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人") 
    

class RolePO(BasePO):
    """
    给用户赋予权限需要通过角色
    """
    __tablename__ = "sys_role"

    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    name = Column("role_name",String(64), unique=True,  comment="角色名")

class permissionPO(BasePO):
    __tablename__ = "sys_permission"

    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True) 
    parentId= Column("parent_id",Integer )
    category= Column("category",Integer )
    name = Column("name",String(64),   comment="名称")
    code = Column("code",String(64), unique=True,  comment="权限为code")
    url = Column("url",String(255)) 
    remark = Column("remark",String(255),   comment="备注")
    
    createUser= Column("create_user", BigInteger,comment="创建人")                   
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人")  

class UserRolePO(BasePO):
    """
    用户_角色表
    """
    __tablename__ = "sys_user_role"

    user= Column("user_id",ForeignKey(f"{UserPO.__tablename__}.{UserPO.id.key}"),   comment="主键id",primary_key=True)
    role = Column("role_id",ForeignKey(f"{RolePO.__tablename__}.{RolePO.id.key}"),   comment="主键id",primary_key=True)
    createUser= Column("create_user", BigInteger,comment="创建人")  
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")   
class permissionRolePO(BasePO):
    """
    权限_角色表
    """
    __tablename__ = "sys_permission_role"

    user= Column("user_id",ForeignKey(f"{UserPO.__tablename__}.{UserPO.id.key}"),   comment="主键id",primary_key=True)
    role = Column("role_id",ForeignKey(f"{RolePO.__tablename__}.{RolePO.id.key}"),   comment="主键id",primary_key=True)
    createUser= Column("create_user", BigInteger,comment="创建人")  
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间") 


class WxMenuPO(BasePO):
    __tablename__ = "wx_menu" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    openId = Column("open_id",String(64),  comment="公众号ID")
    name=Column("name",String(64),  comment="菜名称")
    content= Column("content",String(2048),comment="菜单内容") 
    state=Column("state",Integer,comment="菜单状态") 
    createTime=Column("create_time",DateTime , server_default=func.now() )
    createUser=Column("create_user",BigInteger)
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人") 


class bizDevicePo(BasePO):
    """
    盒子
    """
    __tablename__ = "biz_device" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    boardId = Column("board_id",String(64),comment="盒子唯一标识")
    innerIp=Column("inner_ip",String(64),comment="内部IP")
    ip=Column("ip",String(64),comment="外网IP")
    name=Column("name",String(64),comment="盒子名称")

class bizResourcePO(BasePO):
    __tablename__ = "biz_resource"
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    uid = Column("uuid",String(36),  unique=True,default=uuid.uuid1())
    category = Column("category",Integer,comment="资源类型:0:图片资源,1:视频资源") 
    subCategory = Column("sub_category",Integer,comment="子资源类型")
    deviceId = Column("device_id",ForeignKey(f"{bizDevicePo.__tablename__}.{bizDevicePo.id.key}")) 
    url = Column("url_path",String(255),comment="资源路径,针对根路径下的绝对路径")
    createTime=Column("create_time",DateTime , server_default=func.now() ) 


 
class bizAlarmType(BasePO):
    __tablename__ = "biz_alarm_type"
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    type = Column("type",String(64),unique=True,comment= "告警类型")
    desc= Column("desc",String(128), comment= "告警描述")
    createTime=Column("create_time",DateTime , server_default=func.now() ) 
    updateTime= Column("update_time", DateTime,comment="修改时间") 


class bizAlarmPO(BasePO):
    __tablename__ = "biz_alarm"
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    uuid = Column("uuid",String(64),unique=True,comment= "全局ID盒子上传")
    alarmType= Column("alarm_type",String(64), comment= "告警类型")
   
    videoUid= Column("video_resource_uid",String(36),comment= "视频资源")
    rawImageUid= Column("image_raw_resource_uid",String(36),comment= "原始图片资源")
    markedImageUid= Column("image_marked_resource_uid",String(36),comment= "标注图片资源")
    attachResource1= Column("attach_resource1_uid",String(36),comment= "附加资源1")
    attachResource2= Column("attach_resource2_uid",String(36),comment= "附加资源2")
    attachResource3= Column("attach_resource3_uid",String(36),comment= "附加资源3")
    
    taskSession= Column("task_session_id",String(64),comment= "任务属性")
    taskDesc= Column("task_desc",String(64),comment= "任务描述") 
    alarmTime=Column("alarm_time",DateTime ,comment= "告警事件" )  
    createTime=Column("create_time",DateTime , server_default=func.now() ) 
class bizAlarmAttachPO(BasePO):
    __tablename__ = "biz_alarm_attach" 
    id= Column("alarm_id",ForeignKey(f"{bizAlarmPO.__tablename__}.{bizAlarmPO.id.key}"),comment="主键id",primary_key=True)
    result= Column("result",String(2048),comment="告警Result结果")
    media= Column("media",String(2048),comment="告警media结果")
    gps= Column("gps",String(2048),comment="告警gps结果")









    
    