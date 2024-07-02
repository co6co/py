from model.pos.wx import WxMenuPO
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import or_,and_

class WxMenuFilterItems(absFilterItems): 
	"""
	微信菜单表过滤器
	"""
	name:str=None
	openId:str=None
	
	def __init__(self,name=None): 
		super().__init__(WxMenuPO)
		self.name = name  
	def filter(self)->list:
		"""
		过滤条件
		"""
		filters_arr = []  
		if self.checkFieldValue(self.openId): 
			filters_arr.append(WxMenuPO.openId.__eq__(self.openId))
		if  self.checkFieldValue(self.name):
			filters_arr.append(WxMenuPO.name.like(f"%{self.name}%")) 
		return filters_arr
	def create_List_select():
		raise Exception("为实现")
	
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (WxMenuPO.id.asc(),) 