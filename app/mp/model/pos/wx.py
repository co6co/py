
from co6co_db_ext.po import BasePO ,UserTimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import  relationship,declarative_base
import co6co.utils as tool
from co6co.utils import hash
import sqlalchemy
from sqlalchemy.schema import DDL 
from sqlalchemy import MetaData  
import uuid


class WxMenuPO(UserTimeStampedModelPO):
    __tablename__ = "wx_menu" 
    id = Column("id",Integer,comment="主键",autoincrement=True, primary_key=True)
    openId = Column("open_id",String(64),  comment="公众号ID")
    name=Column("name",String(64),  comment="菜名称")
    content= Column("content",String(2048),comment="菜单内容") 
    state=Column("state",Integer,comment="菜单状态")  