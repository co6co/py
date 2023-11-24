from sqlalchemy import MetaData, INTEGER, Column, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, declarative_base, relationship

# 与属性库实体相关，所有实体集成 BasePO类

metadata = MetaData() 
'''BasePO 得在项目中自建，否则不能创建数据库未找到原因'''
#BasePO = declarative_base(metadata=metadata)