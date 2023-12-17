
from co6co_db_ext.po import BasePO ,metadata,UserTimeStampedModelPO,TimeStampedModelPO
from sqlalchemy import func,INTEGER, Integer,UUID,  INTEGER,BigInteger, Column, ForeignKey, String,DateTime
from sqlalchemy.orm import Relationship 
from sqlalchemy.schema import DDL 
 
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column,relationship
from sqlalchemy import ForeignKey,String,BigInteger
from typing import List


class devicePo(TimeStampedModelPO):
    """
    设备表
    """
    __tablename__="device_table"
    id:Mapped[int] =mapped_column( BigInteger, primary_key=True,autoincrement=True)
    ip:Mapped[str]=mapped_column(String(16)  ,nullable=False,unique=True)
    category:Mapped[int]=mapped_column(INTEGER ,nullable=False ,comment="设备类型")
    category=Column("category_id",ForeignKey(f"device_category.id",ondelete="CASCADE"))
    vendor:Mapped[int]=mapped_column(INTEGER ,nullable=False ,server_default="1", comment="设备产生:1:海康,2:宇视,3:大华")
    name:Mapped[str]=mapped_column(String(64)  )
    code:Mapped[str]=mapped_column(String(32)  )
    categoryPO:Mapped["deviceCategoryPO"]=relationship(back_populates="devicePOs")
    deviceLogPOs:Mapped[List["deviceLogPo"]]=relationship(back_populates="devicePO")
     
    
    def __repr__(self) -> str:
        return f"{self.__class__}->{self.ip},{self.name}"
    
class deviceCategoryPO(TimeStampedModelPO):
    """
    设备类别 
    """
    __tablename__="device_category"
    id:Mapped[int] =mapped_column( BigInteger, primary_key=True )  
    name:Mapped[str]=mapped_column(String(64)  )
    code:Mapped[str]=mapped_column(String(32)  )
    devicePOs:Mapped[List["devicePo"]]=relationship(back_populates="categoryPO")
   
    def __repr__(self) -> str:
        return f"{self.__class__}->{self.id},{self.code},{self.name}"

class deviceLogPo(BasePO):
    """
    设备操作日志
    """
    __tablename__="device_operate_log"
    id:Mapped[int] =mapped_column(BigInteger,primary_key=True,autoincrement=True)
    category:Mapped[str]=mapped_column(String(16),nullable=False,comment="操作类型")
    deviceId:Mapped[int]=mapped_column("device_id",ForeignKey("device_table.id"),nullable=True)
    result:Mapped[str]=mapped_column("result",String(16),nullable=False,comment="操作结果")
    remark:Mapped[str]=mapped_column("remark",String(255),nullable=False,comment="记录操作详情")
    
    devicePO:Mapped["devicePo"]=relationship(back_populates="deviceLogPOs")
    
    def __repr__(self) -> str:
        return f"{self.__class__} md5:{self.category},  {self.result}>"


class sysConfigPO:
    """
    配置相关
    """
    __tablename__="sys_config"
    id:Mapped[int] =mapped_column( BigInteger, primary_key=True )   
    code:Mapped[str]=mapped_column(String(32),nullable=False  )
    name:Mapped[str]=mapped_column(String(64))
    value:Mapped[str]=mapped_column(String(255),nullable=False)
   
    def __repr__(self) -> str:
        return f"{self.__class__} {self.__name__}->{self.ip},{self.name}>"

    

class TasksPO(BasePO):
    __tablename__="user_tasks"
    id= Column("id", BigInteger,comment="主键id",primary_key=True,autoincrement=True)                       #bigint(20) NOT NULL COMMENT '主键id',
    name= Column("name", String(64),comment="名称")                     #varchar(64) DEFAULT NULL COMMENT '名称',
    type= Column("type",Integer,comment="任务类型:0:批量下载任务")                     
    status= Column("status",Integer,comment="状态:0:创建,1:开始ing,2:完成,3:异常结束, 9:取消")     
    data= Column("data",String(2048),comment="数据")     
    createTime= Column("create_time",DateTime,server_default=func.now(),comment="创建时间")     #datetime DEFAULT NULL COMMENT '创建时间', 
    createUser= Column("create_user", BigInteger,comment="创建人")     #bigint(20) DEFAULT NULL COMMENT '创建人',
    