from model.pos.right import UserPO,UserGroupPO
from sqlalchemy .orm.attributes import InstrumentedAttribute 
from typing import Tuple
from co6co_db_ext.db_filter import absFilterItems
from co6co.utils import log
from sqlalchemy import Select, or_,and_

class UserFilterItems(absFilterItems): 
	"""
	用户表过滤器
	"""
	userName:str
	groupName:int
	def __init__(self,userName=None,groupName:str=None): 
		super().__init__(UserPO)
		self.name = userName
		self.groupName = groupName 
		self.listSelectFields=[UserPO.id,  UserPO.state,UserPO.createTime, UserPO.userName ,UserPO.userGroupId,UserGroupPO.name.label("groupName"),UserGroupPO.code.label("groupCode") ]

	def filter(self)->list:
		"""
		过滤条件
		"""
		filters_arr = []  
		if self.checkFieldValue( self.groupName): 
			filters_arr.append(UserGroupPO.name.like(f"%{self.groupName}%"))
		if  self.checkFieldValue(self.name):
			filters_arr.append(UserPO.userName.like(f"%{self.name}%")) 
		return filters_arr
	
	def create_List_select(self):
		select=(
				Select(*self.listSelectFields).join(UserGroupPO,isouter=True) 
				.filter(and_(*self.filter()))  
		) 
		return select 
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		"""
		默认排序
		"""
		return (UserPO.id.asc(),) 