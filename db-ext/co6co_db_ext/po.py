from sqlalchemy import MetaData, INTEGER, Column, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, declarative_base, relationship

# 与属性库实体相关，所有实体集成 BasePO类

metadata = MetaData() 
BasePO = declarative_base(metadata=metadata)