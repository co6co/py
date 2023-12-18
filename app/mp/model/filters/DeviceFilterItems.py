from model.enum import device_type
from model.pos.biz import bizDevicePo
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import or_,and_,Select



class CameraFilterItems(absFilterItems): 
	"""
	ip相机
	"""
	name:str 
	def __init__(self ): 
		super().__init__(bizDevicePo) 
		self.listSelectFields=[bizDevicePo.id,bizDevicePo.name,bizDevicePo.createTime,bizDevicePo.ip,bizDevicePo.innerIp]

	def filter(self)->list:
		"""
		过滤条件
		"""
		filters_arr = []  
		filters_arr.append(bizDevicePo.deviceType==device_type.ip_camera.val) 
		if  self.checkFieldValue(self.name):
			filters_arr.append(bizDevicePo.name.like(f"%{self.name}%")) 
		return filters_arr
	def create_List_select(self):
		select=(
				Select(*self.listSelectFields)#.join(device.deviceCategoryPO,isouter=True)
				.filter(and_(*self.filter())) 
				.limit(self.limit).offset(self.offset)
		) 
		return select
	
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (bizDevicePo.id.asc(),) 