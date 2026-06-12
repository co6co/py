

from typing import Optional

from sanic import Request
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select 

from co6co_db_ext.db_utils import db_tools 
from ..base_view import AuthMethodView
from ..biz_view import AbsPkView
from ...model.filters.dict_filter import DictFilter
from ...model.pos.other import sysDictPO, sysDictTypePO
from ...model.enum import dict_state

class DictSelectView(AuthMethodView):
    routePath = "/select"
    @property
    def dictTypeCode(self):
        return self.json.get("dictTypeCode")
    @property
    def dictTypeId(self):
        return self.json.get("dictTypeId")
    @property
    def parentId(self):
        return self.json.get("parentId")
     
    @property
    def category(self):
        return self.json.get("category") 
    async def getParentCondition(self):  
        if self.parentId is None or not isinstance(self.parentId, int): 
            return  sysDictPO.parentId==None
        return sysDictPO.parentId.__eq__(self.parentId)
        
                
    async def post(self ):
        """ 
        获取字典选择
        dictTypeCode: 字典类型代码
        """
        # NameValueFlag = 0,
        # NameValue = 1,
        # NameFlag = 2,
        # All = 999,
        FIELD_MAP = {
            1: [sysDictPO.id, sysDictPO.name, sysDictPO.value],
            2: [sysDictPO.id, sysDictPO.name, sysDictPO.flag],
            999: [sysDictPO.id, sysDictPO.name, sysDictPO.flag, sysDictPO.value, sysDictPO.desc],
        }
        DEFAULT_FIELDS = [sysDictPO.id, sysDictPO.name, sysDictPO.flag, sysDictPO.value]

        fields = FIELD_MAP.get(self.category, DEFAULT_FIELDS) 
        # 2. 统一构建 where 条件
        where = [sysDictPO.state.__eq__(dict_state.enabled.val),await self.getParentCondition()]  
        # 3. 根据 dictTypeId 直接决定查询方式 
        if self.dictTypeId:
            where.append(sysDictPO.dictTypeId.__eq__(self.dictTypeId))
            query = Select(*fields).filter(*where)
        else:
            where.append(sysDictTypePO.code.__eq__(self.dictTypeCode))
            query = (
                Select(*fields)
                .join(sysDictTypePO, onclause=sysDictPO.dictTypeId == sysDictTypePO.id)
                .filter(*where)
            )

        query = query.order_by(sysDictPO.order.asc())
        return await self.query_list(query, isPO=False)


class Views(AuthMethodView):
    async def get(self ):
        """
        字典、字典类型状态
        枚举类型 : dict_state
        """
        return self.response_json(Result.success(data=dict_state.to_dict_list()))

    async def post(self ):
        """
        table数据 
        """
        param = DictFilter()
        param.__dict__.update(self.json)
        return await self.query_tree(param.create_List_select(),pid_field='parentId')

    async def put(self  ):
        """
        增加
        """
        po = sysDictPO()
        userId = self.userId

        async def before(po: sysDictPO, session: AsyncSession, request):
            exist = await db_tools.exist(session, sysDictPO.dictTypeId == po.dictTypeId, sysDictPO.value == po.value,   column=sysDictPO.id)
            if exist:
                return Result.fail(message=f"值'{po.value}'在该字典中已存在！")
        return await self.add( po, userId=userId, beforeFun=before)

class DictOneView(AuthMethodView):
    routePath = "/one/<pk:int>"
    async def get(self ):
        """
        获取字典详情
        """
        id=self.match_info.get("pk")
       
        select=Select(sysDictPO).where(sysDictPO.id.__eq__(id))
        poDict =await   self.actuator.query_one_mappings(select)
        if poDict is None:
            return self.response_json(Result.fail(message=f"字典不存在！"))
        else:
            return self.response_json( Result.success(data=poDict) )
            
class View(AbsPkView): 
    async def put(self ):
        """
        编辑
        """
        async def before(oldPo: sysDictPO, po: sysDictPO, session: AsyncSession,*args,**kwargs):
            exist = await db_tools.exist(session, sysDictPO.dictTypeId == po.dictTypeId, sysDictPO.value == po.value, sysDictPO.id != oldPo.id, column=sysDictPO.id)
            if exist:
                return Result.fail(message=f"'{po.value}'在该字典中已存在！")

        return await self.edit(self.routeValue, sysDictPO, userId=self.userId, fun=before)

    async def delete(self):
        """
        删除
        """
        return await self.remove(self.routeValue, sysDictPO)
