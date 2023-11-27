from model.pos.right import UserPO
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import or_,and_

class UserFilterItems(absFilterItems): 
	"""
	用户表过滤器
	"""
	def __init__(self,userName=None,role_id:int=None): 
		super().__init__(UserPO)
		self.name = userName
		self.roleId = role_id 

	def filter(self)->list:
		"""
		过滤条件
		"""
		filters_arr = []  
		if self.checkFieldValue( self.roleId): 
			filters_arr.append(UserPO.roleId.__eq__(self.roleId))
		if  self.checkFieldValue(self.name):
			filters_arr.append(UserPO.userName.like(f"%{self.name}%")) 
		return filters_arr
	
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (UserPO.id.asc(),) 