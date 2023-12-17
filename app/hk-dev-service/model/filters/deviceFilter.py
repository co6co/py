from model.pos import device
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple,List
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import func,or_,and_,Select 


class DeviceFilterItems(absFilterItems): 
	"""
	用户表过滤器
	"""
	name:str
	listSelectFields: List[InstrumentedAttribute]
	def __init__(self ): 
		super().__init__(device.devicePo) 
		self.listSelectFields=[	
                        device.devicePo.id,device.devicePo.ip,device.devicePo.name,device.devicePo.code,
                        device.devicePo.createTime, 
                        device.deviceCategoryPO.name.label("categoryName"),device.deviceCategoryPO.code.label("categoryCode")
                        ]

	def filter(self)->list:
		"""
		过滤条件 
		"""
		filters_arr = []   
		if self.checkFieldValue(self.name):
			filters_arr.append(device.devicePo.name.like(f"%{self.name}%"))
		return filters_arr

	def create_List_select(self):
		select=(
				Select(*self.listSelectFields).join(device.deviceCategoryPO,isouter=True)
				.filter(and_(*self.filter())) 
				.limit(self.limit).offset(self.offset)
		) 
		return select
 
	
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (device.devicePo.id.asc(),) 