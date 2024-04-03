
from co6co_db_ext.po import BasePO ,UserTimeStampedModelPO,TimeStampedModelPO
from sqlalchemy import func,INTEGER, SMALLINT,Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import  relationship,declarative_base,Relationship
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL 
from sqlalchemy import MetaData  
from sqlalchemy.dialects.mysql import VARCHAR


class WxMenuPO(UserTimeStampedModelPO):
    __tablename__ = "wx_menu" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    name=Column("name",String(64),  comment="菜名称")
    content= Column("content",String(2048),comment="菜单内容") 
    state=Column("state",Integer,comment="菜单状态") 
    openId = Column("open_id",String(64),  comment="公众号ID")  

class WxUserPO(TimeStampedModelPO):
    __tablename__ = "wx_user" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    accountUid=Column("account_uid",ForeignKey(f"sys_account.uuid",ondelete="CASCADE"),nullable=False,index=True ) 
    openid = Column("open_id",String(64),  comment="用户OpenID",nullable=False) 
    nickName=Column("nick_name",String(255).with_variant(VARCHAR(255, charset="utf8mb4"), "mysql", "mariadb"), comment="nick_name"   )
    sex=Column("sex",SMALLINT,comment="0:男,1:女")
    language=Column("language",String(8),comment="语言")
    city=Column("city",String(16),comment="城市")
    province=Column("province",String(16),comment="城市")
    country=Column("country",String(64),comment="国家")
    headimgUrl=Column("headimg_url",String(255),comment="图像URL")
    privilege=Column("privilege",String(255),comment="特权") 
    ownedAppid = Column("owned_app_id",String(64),comment="公众号appId",nullable=False)
    accountPO=Relationship("AccountPO",back_populates="wxUserPO")


