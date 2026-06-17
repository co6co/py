from co6co_db_ext.actuator import Actuator
from sanic import Request
from co6co_db_ext.db_utils import db_tools 
from co6co_db_ext.db_filter import absFilterItems

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session
from typing import TypeVar, TypeAlias, Dict, List, Any, Tuple, Optional, Callable, Awaitable
from co6co.data.result import Result, Page_Result

from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm.attributes import InstrumentedAttribute

from sqlalchemy import Select,  Update, Delete, Insert
from co6co_sanic_ext .view_model import BaseView, BaseClsView
from co6co_db_ext.po import BasePO
from co6co.utils.tool_util import list_to_tree, get_current_function_name


from co6co_web_session.base import SessionDict
from multiprocessing.managers import DictProxy


from co6co.utils import log
from ..model.params import associationParam
from co6co.utils.modules import deprecated
from co6co_db_ext.appconfig import AppConfig
from ..services import get_db_session, set_rollback, get_cache




TypeOneFun: TypeAlias = Callable[[
    BasePO, AsyncSession, Request], Awaitable[None | Any]]
TypeTwoFun: TypeAlias = Callable[[
    BasePO, BasePO, AsyncSession, Request], Awaitable[None | Any]]
TypeListFun: TypeAlias = Callable[[
    List[BasePO], AsyncSession, Request], Awaitable[None | Any]]
# 关联的创建
TypeCreateFun: TypeAlias = Callable[[  AsyncSession, int | str], Awaitable[BasePO]]


async def get_one(request: Request, select: Select, isPO: bool = True):
    """
    获取一条记录
    """
    # call = QueryOneCallable(get_db_session(request))
    # return await call(select, isPO)

    actuator = Actuator(get_db_session(request))
    if isPO:
        return await actuator.query_one_entity(select)
    else:
        return await actuator.query_one_mappings(select)

 

def peraseRequest(request: Request) -> Tuple[AsyncSession, SessionDict, DictProxy]:

    session: AsyncSession = get_db_session(request)
    memSession: SessionDict = request.ctx.Session  # 需要知道这个是哪的
    cache: DictProxy = get_cache(request)
    return session, memSession, cache


class BaseDbClsView(BaseClsView):
    """
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用 
    """
    @property
    def db_session(self) -> AsyncSession | scoped_session:
        return get_db_session(self.request)

    @property
    def shared_cache(self) -> DictProxy:
        return self.app.shared_ctx.cache

    @property
    def app_config(self) -> AppConfig:
        if not hasattr(self, '_app_config') or self._app_config is None:
            self._app_config = AppConfig.get_config(self.config)
        return self._app_config

    @property
    def actuator(self):
        if hasattr(self, '_actuator'):
            return self._actuator
        self._actuator = Actuator(self.db_session)
        return self._actuator

    
    def response_error(self,  e: Exception, msg="执行不成功,错误码：{}"):
        """
        响应错误 message
        """ 
        return self.response_json(self.actuator.error(e,msg))

    async def update_one(self, select:Select, edit:Callable[[BasePO], Result|None]):
        """
        更新一条记录
        """
        try:
            po = await self.actuator.query_one_entity(select)
            if po is None:
                return self.response_json(Result.fail(message="未查询到数据"))
            result = edit(  po)
            if result is not None: 
                return self.response_json(result)
            return self.response_json(Result.success(result))
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="更新失败{}")
    async def get_one(self, select: Select, isPO: bool = True, remove_db_instance: bool = True, resultHanlder: Callable[[Any], Any] = None):
        """
        从数据库中获取一个对象
        resultHanlder: 不为空是，返回值将作为最终的返回结果
              使有机会改变从数据库中查询的结果              
        """
        # call = QueryOneCallable(self.db_session)
        # result = await call(select, isPO)
        result = await get_one(self.request, select, isPO)
        if isPO and remove_db_instance:
            result = Actuator.remove_db_instance_state(result)
        if resultHanlder is not None:
            bckResult = resultHanlder(result)
            if bckResult is not None:
                result = bckResult
        if result is None:
            return self.response_json(Result.fail(message="未查询到数据"))
        else:
            return self.response_json(Result.success(result))

    def set_rollback(self):
        """
        设置回滚
        """
        set_rollback(self.request, True)

    

    async def query_mapping(self, select: Select, oneRecord: bool = False):
        """
        执行查询: 一个列表|一条记录
        """
        try:
            if oneRecord:
                result = await self.actuator.query_one_mappings(select)
                return self.response_json(Result.success(result))
            else:
                result = await self.actuator.query_all_mappings(select)
                return self.response_json(Result.success(result))
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="查询失败{}")

    async def _query(self,  select: Select, isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询: 列表 
        """
        try:
            if isPO:
                result = await self.actuator.query_all_entity_mappings(select)
            else:
                result = await self.actuator.query_all_mappings(select)
            if remove_db_instance:
                result = Actuator.remove_db_instance_state(result)
            return result 
        except Exception as e:
            self.set_rollback()
            self.response_error(e, msg="_query 失败{}")
            return None

    async def exist(self,   *filters: ColumnElement[bool], column: InstrumentedAttribute = "*"):
        """
        查看对象是否操作
        """
        self.actuator.exist(*filters, column=column)

    async def query_list(self, select: Select,   isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询:  列表 
        """
         
        result = await self._query(select,   isPO, remove_db_instance, param)
        return self.response_json(Result.success(data=result))
         

    async def query_tree(self,   select: Select, rootValue: any = None, pid_field: str = "pid", id_field: str = "id", isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询: tree列表 
        """
        try:
            result = await self._query(select,   isPO, remove_db_instance, param)
            if result is None:
                treeList = []
            else:
                treeList = list_to_tree(result, rootValue, pid_field, id_field)
            if len(treeList) == 0:
                return self.response_json(Result.success(data=[]))
            return self.response_json(Result.success(data=treeList))
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="查询失败{}")

    async def query_page(self,   filter: absFilterItems, isPO: bool = True, remove_db_instance=True):
        """
        分页查询
        """
        filter.__dict__.update(self.request.json)
        try:
            #from co6co.utils import log
            #log.warn(f"query_page,filter  isPO:{isPO},remove_db_instance:{remove_db_instance}")
            pageList = await self.actuator.query_page(filter)
            #query = QueryPagedByFilterCallable(self.db_session)
            #total, result = await query(filter, isPO, remove_db_instance)
            #pageList = Page_Result.success(result, total=total)
            return self.response_json(pageList)
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="查询失败{}")

    async def execSqls(self,   *sml: Update | Delete | Insert, callBck=None, smlParamList: List[Dict | Tuple | List] = None): 
        try:
            result = []
            index = 0
            for sql in sml:
                param = None
                if smlParamList is not None and len(smlParamList) == len(sml):
                    param = smlParamList[index]
                r = await self.actuator.execSQL(sql, param)
                result.append(r)
                index += 1
            if callBck is not None:
                return await callBck(*result)
        except Exception as e:
            self.set_rollback()
            return self.response_error(e)
                
    def check_backResult(self, result: Result|bool):
       reslt=self.actuator._check_backResult(result)
       if reslt is not None:
           return self.response_json(reslt)
       return None
    
    async def batchAdd(self, poList: List[BasePO], userId=None, beforeFun: TypeOneFun = None, afterFun: TypeListFun = None):
        try: 
            for po in poList:
                po.add_assignment(userId)
                if beforeFun is not None:
                    result = await beforeFun(po, self.actuator.session, self.request)
                    result=self.check_backResult(result) 
                    if result is not None:
                        self.set_rollback()
                        return result
            self.actuator.add_all(*poList)
            await self.actuator.session.flush()
            if afterFun is not None:
                
                result=await afterFun(poList, self.actuator.session, self.request)
                result=self.actuator._check_backResult(result) 
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            return self.response_json(Result.success()) 
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="批量增加失败{}")

    async def add(self,   po: BasePO, json2Po: bool = True, userId=None, beforeFun: TypeOneFun = None, afterFun: TypeOneFun = None):
        """
        增加 

        request: Request, 
        po: BasePO,      #实体类对象 
        userId=None, # 用户ID
        beforeFun(po, session, request),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(po, session, request),     # 可在实体中获取 自增id

        return JSONResponse
        """
        try:
            if json2Po:
                po.__dict__.update(self.request.json) 
            po.add_assignment(userId)

            if beforeFun is not None:
                result = await beforeFun(po,self.actuator. session, self.request)
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            self.actuator.add_all(po)
            await self.actuator.session.flush()
            if afterFun is not None: 
                result=await afterFun(po, self.actuator.session, self.request)
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            return self.response_json(Result.success()) 
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="增加失败{}")

    async def edit(self,   pkOrSelect:  int | str | Select, poType: TypeVar, po: Optional[BasePO] = None, userId=None, fun: TypeOneFun = None, json2Po: bool = True):
        """
        编辑

        pk: any,          # 主键
        poType: TypeVar,  # 实体类型
        po:BasePO    ,    # None:根据传入的 poType创建,用 request.json赋值
        userId=None, # 用户ID
        fun=None,    # 执行一些其他操作，返回值将直接返回客户端并且回滚数据库操作
        json2Po:bool # 根据 请求的json 转换的对象更新 实体对象，在 fun 之前执行

        return JSONResponse
        """
        try:
            if po is None:
                po = poType()
                po.__dict__.update(self.request.json)
            oldPo: BasePO = None
            if isinstance(pkOrSelect, Select):
                oldPo =  await db_tools.execForPo(self.actuator.session, pkOrSelect, remove_db_instance_state=False)
            else:
                oldPo: BasePO = await self.actuator.session.get_one(poType, pkOrSelect)
            if oldPo is None:
                return self.response_json(Result.fail(message=f"未查到‘{pkOrSelect}’对应的信息!"))
            oldPo.edit_assignment(userId)
            if json2Po:
                oldPo.update(po)
            if fun is not None:
                result = await fun(oldPo, po, self.actuator.session, self.request)
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            return self.response_json(Result.success())
           
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="更新失败{}")

    async def remove(self,   pk: any, poType: TypeVar,  beforeFun: TypeOneFun = None, afterFun: TypeOneFun = None):
        """
        删除   
        pk: any,      #主键值
        poType: TypeVar # 实体类型
        beforeFun(oldPo, session),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(oldPo, session, request),     # 返回值将直接返回客户端，回滚数据库操作

        return JSONResponse
        """
        try:
            
            oldPo: BasePO = await self.actuator.session.get_one(poType, pk)
            if oldPo == None:
                return self.response_json(Result.fail(message=f"未找到‘{pk}’对应的信息!"))
            if beforeFun != None:
                result = await beforeFun(oldPo,self.actuator.session, self.request)
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            await self.actuator.session.delete(oldPo)
            if afterFun != None:
                result = await afterFun(oldPo, self.actuator.session, self.request)
                result=self.check_backResult(result) 
                if result is not None:
                    self.set_rollback()
                    return result
            return self.response_json(Result.success())
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="删除失败{}")

    def get_associationParam(self):
        param = associationParam()
        param.__dict__.update(self.json)
        return param

    async def save_association(self, currentUser: int, delSml: Delete, createPo: TypeCreateFun = None, param: associationParam = None, delSmlParam: Dict | List | Tuple = None):
        """(
        保存关联菜单
        delSml:Delete 删除语句
        createPo:(session,id)=>basePO
        """
        if param is None:
            param = self.get_associationParam() 
        try:
            isChanged = False
            # 移除
            if (param.remove is not None and len(param.remove) > 0):
                # Delete(userGroupProjectPO).filter(userGroupProjectPO.projectId ==self.projectId, userGroupProjectPO.userGroupId .in_(bindparam("remove")))
                result = await self. actuator .execSQL(delSml, delSmlParam)
                if result > 0:
                    isChanged = True
            # 增加
            if (param.add is not None and len(param.add) > 0):
                addpoList = []
                for id in param.add:
                    po: BasePO = await createPo(self.actuator.session, id)
                    po.add_assignment(currentUser)
                    addpoList.append(po)
                if len(addpoList) > 0:
                    isChanged = True
                    self. actuator.add_all(*addpoList)
            if isChanged:
                return self.response_json(Result.success())
            else:
                return self.response_json(Result.fail(message="未改变"))
        except Exception as e:
            self.set_rollback()
            return self.response_error(e, msg="保存失败{}") 


@deprecated("该类已废弃，请使用 BaseDbView 类替代")
class BaseMethodView(BaseView):
    """
    # 以不在使用该类, 方法已移动到BaseDbView中
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用

    views.POST  : --> query list
    views.PUT   :---> Add 
    view.PATCH    :---> Edit
    view.DELETE :---> del

    """

    def get_db_session(self, request: Request) -> AsyncSession | scoped_session:
        return BaseDbClsView(request).db_session

    def get_shared_Cache(self, request: Request) -> DictProxy:
        return BaseDbClsView(request).shared_cache
        
    def response_error(self, request: Request, e: Exception):
        """
        响应错误 message
        """
        return BaseDbClsView(request).response_error(e)

    async def get_one(self, request: Request, select: Select, isPO: bool = True, remove_db_instance: bool = True, resultHanlder: Callable[[Any], Any] = None):
        """
        从数据库中获取一个对象
        resultHanlder: 不为空是，返回值将作为最终的返回结果
              使有机会改变从数据库中查询的结果              
        """
        view = BaseDbClsView(request)
        return await view.get_one(select, isPO, remove_db_instance, resultHanlder)

 

    async def query_mapping(self, request: Request, select: Select, oneRecord: bool = False):
        """
        执行查询: 一个列表|一条记录
        """
        view = BaseDbClsView(request)
        return await view.query_mapping(select,  oneRecord)

    async def exist(self, request: Request,  *filters: ColumnElement[bool], column: InstrumentedAttribute = "*"):
        """
        查看对象是否操作
        """
        view = BaseDbClsView(request)
        return await view.exist(*filters,  column=column)

    async def query_list(self, request: Request, select: Select,   isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询:  列表 
        """
        view = BaseDbClsView(request)
        return await view.query_list(select,   isPO, remove_db_instance, param)

    async def query_tree(self, request: Request, select: Select, rootValue: any = None, pid_field: str = "pid", id_field: str = "id", isPO: bool = True, remove_db_instance: bool = True, param: Dict | List | Tuple = None):
        """
        执行查询: tree列表 
        """
        view = BaseDbClsView(request)
        return await view.query_tree(select,  rootValue, pid_field, id_field, isPO, remove_db_instance, param)

    async def query_page(self, request: Request, filter: absFilterItems, isPO: bool = True, remove_db_instance=True):
        """
        分页查询
        """
        view = BaseDbClsView(request)
        return await view.query_page(filter, isPO, remove_db_instance)

    async def execSqls(self, request: Request, *sml: Update | Delete | Insert, callBck=None, smlParamList: List[Dict | Tuple | List] = None):
        view = BaseDbClsView(request)
        return await view.execSqls(*sml, callBck, smlParamList)

    async def batchAdd(self, request: Request, poList: List[BasePO], userId=None, beforeFun: TypeOneFun = None, afterFun: TypeListFun = None):
        view = BaseDbClsView(request)
        return await view.batchAdd(poList, userId, beforeFun, afterFun)

    async def add(self, request: Request, po: BasePO, json2Po: bool = True, userId=None, beforeFun: TypeOneFun = None, afterFun: TypeOneFun = None):
        """
        增加 

        request: Request, 
        po: BasePO,      #实体类对象 
        userId=None, # 用户ID
        beforeFun(po, session, request),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(po, session, request),     # 可在实体中获取 自增id

        return JSONResponse
        """
        view = BaseDbClsView(request)
        return await view.add(po, json2Po, userId, beforeFun, afterFun)

    async def edit(self, request: Request, pkOrSelect:  int | str | Select, poType: TypeVar, po: Optional[BasePO] = None, userId=None, fun: TypeOneFun = None, json2Po: bool = True):
        """
        编辑

        request: Request, 
        pk: any,          # 主键
        poType: TypeVar,  # 实体类型
        po:BasePO    ,    # None:根据传入的 poType创建,用 request.json赋值
        userId=None, # 用户ID
        fun=None,    # 执行一些其他操作，返回值将直接返回客户端并且回滚数据库操作
        json2Po:bool # 根据 请求的json 转换的对象更新 实体对象，在 fun 之前执行

        return JSONResponse
        """
        view = BaseDbClsView(request)

        return await view.edit(pkOrSelect, poType, po, userId, fun, json2Po)

    async def remove(self, request: Request, pk: any, poType: TypeVar,  beforeFun: TypeOneFun = None, afterFun: TypeOneFun = None):
        """
        删除  
        request: Request, 
        pk: any,      #主键值
        poType: TypeVar # 实体类型
        beforeFun(oldPo, session),    # 执行一些其他操作，返回值将直接返回客户端，回滚数据库操作
        afterFun(oldPo, session, request),     # 返回值将直接返回客户端，回滚数据库操作

        return JSONResponse
        """
        view = BaseDbClsView(request)
        return await view.remove(pk, poType, beforeFun, afterFun)

    async def save_association(self, request: Request, currentUser: int, delSml: Delete, createPo: TypeCreateFun, param: associationParam = None, delSmlParam: Dict | List | Tuple = None):
        """
        保存关联菜单
        delSml:Delete 删除语句
        createPo:(id)=>basePO
        """
        view = BaseDbClsView(request)
        return await view.save_association(currentUser, delSml, createPo, delSmlParam, param)


"""
class AuthMethodView(BaseMethodView): 
   decorators=[authorized]  
    def update():
         stmt = (
            Update(BoatGroupPO)
            .where(BoatGroupPO.groupType==Device_Group_type.site.key)
            .values(priority=99999)
        ) 
        return await session.execute(stmt) 
"""
