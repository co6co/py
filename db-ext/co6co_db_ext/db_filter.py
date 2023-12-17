
from abc import ABC, abstractclassmethod 
from .page_param import Page_param
from typing import TypeVar,Tuple,List,Dict,Any,Union,Iterator
from sqlalchemy .orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy import func,or_,and_,Select  

class absFilterItems(ABC, Page_param): 
	"""
	抽象过滤器
	配合 DbOperations 使用
	"""
	listSelectFields: List[InstrumentedAttribute]
	def __init__(self,po:TypeVar) -> None:
		super().__init__() 

		self.po_type=po # 与排序相关
		pass

	@property
	def offset(self):
		return self.get_db_page_index()*self.pageSize
	@property
	def limit(self):
		return self.pageSize
	

	@abstractclassmethod
	def filter(self)->List[ColumnElement[bool]]:
		raise NotADirectoryError("Can't instantiate abstract clas") 
	def _getOrderby(self)->List[Dict[str,str]]: 
		if self.orderBy and  self.order:
			by=self.orderBy.split(",")
			order=self.order.split(",")
			if len(by)==len(order):
				return [{b:o} for b in by for o in order if b and o]
			else:
				return [{b:order[0]} for b in by if b ]
		elif self.orderBy:
			by=self.orderBy.split(",")
			return [{b:"asc"} for b in by if b ]
		return [] 
	
	@abstractclassmethod
	def getDefaultOrderBy(self)->Tuple[InstrumentedAttribute]:
		raise NotADirectoryError("Can't instantiate abstract clas") 
	 
	def getOrderBy(self)->List[InstrumentedAttribute]:
		""" 
		获取排序规则
		
		"""
		orderList= self._getOrderby()
		if len(orderList)==0: return self.getDefaultOrderBy()
		# () 获取的结果不能重复 取*
        # []  获取的结果能重复 取*
		else: return [ self. po_type.__dict__[key].desc() if it[key] and it[key].lower()=="desc" else  self. po_type.__dict__[key].asc()  for it in orderList for key in it.keys()]
	
	def checkFieldValue(self,fielValue:Any)->bool:
		if type(fielValue) == str and fielValue:return True
		if type(fielValue) == int :return True
		if type(fielValue) == bool :return True
		return False

	@abstractclassmethod
	def create_List_select(self):
		select=(
				Select(*self.listSelectFields)#.join(device.deviceCategoryPO,isouter=True)
				.filter(and_(*self.filter())) 
				.limit(self.limit).offset(self.offset)
		) 
		return select
	
	@property
	def list_select(self):
		return self.create_List_select().limit(self.limit).offset(self.offset) 
	@property
	def count_select(self): 
		return Select( func.count( )).select_from(self.create_List_select())
