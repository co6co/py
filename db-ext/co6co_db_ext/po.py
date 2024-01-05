from sqlalchemy import func,MetaData, INTEGER, Column, ForeignKey, String,BigInteger,DateTime
from sqlalchemy.orm import  DeclarativeBase, declarative_base, relationship,QueryPropertyDescriptor,Query


# 与属性库实体相关，所有实体集成 BasePO类

metadata = MetaData()  
#class BasePO(declarative_base(metadata=metadata)):  
class BasePO(DeclarativeBase):
    __abstract__=True
    @property
    def query()->Query:
        return None
    def to_dict(self):
        return {c.name: getattr(self, c.name, None) for c in self.__table__.columns}
    
class TimeStampedModelPO(BasePO):
    __abstract__=True
                 
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")
    updateTime= Column("update_time", DateTime,comment="更新时间") 
class UserTimeStampedModelPO(TimeStampedModelPO):
    __abstract__=True

    createUser= Column("create_user", BigInteger,comment="创建人")  
    updateUser= Column("update_user", BigInteger,comment="修改人")   
