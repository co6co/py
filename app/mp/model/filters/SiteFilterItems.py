from model.pos.biz import bizSitePo,bizBoxPO,bizCameraPO,bizRouterPO
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import absFilterItems
from co6co.utils import log
from sqlalchemy import Select, or_,and_
from sqlalchemy.orm import joinedload, contains_eager,selectinload

class SiteDiveceFilterItems(absFilterItems): 
	"""
	Site表过滤器
	"""
	cameraName:str=None
	def __init__(self ): 
		super().__init__(bizSitePo)
		self.listSelectFields=[bizSitePo.id,bizSitePo.name]

	def filter(self)->list:
		"""
		过滤条件
		"""
		filters_arr = []  
		if self.checkFieldValue( self.cameraName): 
			filters_arr.append(bizSitePo.name.like(self.cameraName)) 
		return filters_arr
	def create_List_select(self):
		select=(
				Select(bizSitePo).options(selectinload(bizSitePo.boxPO))
				#.options(joinedload(bizSitePo.boxPO))  
				.options(joinedload(bizSitePo.camerasPO)) 
				.filter(and_(*self.filter()))  
		) 
		return select 
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (bizSitePo.id.asc(),) 