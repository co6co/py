from co6co_db_ext.po import BasePO ,TimeStampedModelPO,UserTimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import  relationship,declarative_base,Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL 
from sqlalchemy import MetaData  
import uuid
from model.pos.wx import WxUserPO  #AccountPO 引用到

#metadata = MetaData() 
#BasePO = declarative_base(metadata=metadata)
class UserPO(UserTimeStampedModelPO):
    __tablename__ = "sys_user" 
    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    userName = Column("user_name",String(64), unique=True,  comment="userName")
    category= Column("category",Integer,)
    password = Column("user_pwd",String(256),comment="密码")
    salt=Column("user_salt",String(64),comment="pwd=盐")
    userGroupId=Column("user_group_id",ForeignKey(f"sys_user_group.id",ondelete="CASCADE"))
    state=Column("state",INTEGER, comment="用户状态:0启用,1锁定,2禁用",server_default='0', default=0)  
    avatar=Column("avatar",String(256),comment="图像URL")
    remark=Column("remark",String(1500),comment="备注") 
     
    accountPOs=Relationship("AccountPO",back_populates="userPO",uselist=True,passive_deletes=True)
    userGroupPO=Relationship("UserGroupPO",back_populates="userPOs") 
    rolePOs=Relationship("RolePO",secondary="sys_user_role",back_populates="userPOs",passive_deletes=True)
    
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
        return {"id":self.id,"userName":self.userName, "group_id":self.userGroupId}
 
    
class AccountPO(BasePO):
    """
    账号：以各种方式登录系统 用户名  EMAIL openID
    """
    __tablename__ = "sys_account" 
    uid = Column("uuid",String(36),  primary_key=True,default=uuid.uuid1()) 
    userId=Column("user_id",ForeignKey(f"{UserPO.__tablename__}.{UserPO.id.name}",ondelete="CASCADE"),nullable=False,index=True,unique=True)
    accountName = Column("account_name",String(64), unique=True)
    attachInfo= Column("attach_info",String(255),comment="有些信息需要存下来，才能配合账号使用")
    category = Column("category",INTEGER)
    status = Column("status",String(16),comment="状态,根据账号类型，有不同得解释")
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")
    createUser= Column("create_user", BigInteger,comment="创建人")                   
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人") 

    userPO=Relationship(UserPO,back_populates="accountPOs")
    wxUserPO=Relationship("WxUserPO",back_populates="accountPO",uselist=False,passive_deletes=True)


    

class UserGroupPO(BasePO):
    """
    用户组
    """
    __tablename__ = "sys_user_group" 
    id = Column("id",Integer,autoincrement=True, primary_key=True)
    name = Column("group_name",String(64) ) 
    code = Column("group_code",String(64), unique=True) 
    createUser= Column("create_user", BigInteger,comment="创建人")  
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")                 
    updateTime= Column("update_time", DateTime,comment="修改时间") 
    updateUser= Column("update_user", BigInteger,comment="修改人") 
    
    userPOs=Relationship("UserPO",back_populates="userGroupPO",uselist=True,passive_deletes=True) 

class RolePO(TimeStampedModelPO):
    """
    给用户赋予权限需要通过角色
    """
    __tablename__ = "sys_role"

    id = Column("id",BigInteger,comment="主键",autoincrement=True, primary_key=True)
    name = Column("role_name",String(64) ,comment="角色名") 
    code = Column("role_code",String(64), unique=True) 
    userPOs=Relationship("UserPO",secondary="sys_user_role",back_populates="rolePOs",passive_deletes=True)
    permissionPOs=Relationship("permissionPO",secondary="sys_permission_role",back_populates="rolePOs",passive_deletes=True)

class UserRolePO(UserTimeStampedModelPO):
    """
    用户_角色表
    """
    __tablename__ = "sys_user_role"

    user= Column("user_id",ForeignKey(f"{UserPO.__tablename__}.{UserPO.id.name}"),   comment="主键id",primary_key=True)
    role = Column("role_id",ForeignKey(f"{RolePO.__tablename__}.{RolePO.id.name}"),   comment="主键id",primary_key=True)


class permissionPO(UserTimeStampedModelPO):
    __tablename__ = "sys_permission"

    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True) 
    parentId= Column("parent_id",Integer )
    category= Column("category",Integer )
    name = Column("name",String(64),   comment="名称")
    code = Column("code",String(64), unique=True,  comment="权限为code")
    url = Column("url",String(255)) 
    remark = Column("remark",String(255),   comment="备注")

    rolePOs=Relationship("RolePO",secondary="sys_permission_role",back_populates="permissionPOs",passive_deletes=True)
    


class permissionRolePO(UserTimeStampedModelPO):
    """
    权限_角色表
    """
    __tablename__ = "sys_permission_role"

    permission= Column("user_id",ForeignKey(f"{permissionPO.__tablename__}.{permissionPO.id.name}",ondelete="CASCADE"),   comment="主键id",primary_key=True)
    role = Column("role_id",ForeignKey(f"{RolePO.__tablename__}.{RolePO.id.name}",ondelete="CASCADE"),   comment="主键id",primary_key=True)
    createUser= Column("create_user", BigInteger,comment="创建人")  
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间") 
    