from optparse import Option
from typing import TypeVar, Type, Generic, Tuple
from co6co.data.result import Result, Page_Result
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.session import AsyncSession
from .base_view import AuthMethodView
from sqlalchemy import Select, Delete
from abc import ABC, abstractmethod
from co6co_db_ext.po import BasePO
from sqlalchemy.sql.elements import ColumnElement
from co6co_db_ext.actuator import OperationOption
from .aop.response_apo import response

FilterType = TypeVar("FilterType", bound=absFilterItems)
FilterClass = Type[FilterType]


class AbsQueryView(AuthMethodView):
    """
    基础查询视图
    需要传入absFilterItems 子类及类属性cls
    HTTPMethodView 阻断了阻断 Generic 的设置流程，导致 __parameters__不存在
    所有不能使用 AbsQueryView(AuthMethodView, Generic[FilterType]):

    现在不用泛型继承,[感觉 Generic[FilterType]，也没意义,以为它也需要 cls 属性，所有只用类属性]
    使用示例：
    class ProjectQueryView(AbsQueryView[Filter]):
        route = "/query"
        cls = Filter
    """

    routePath = "/query"
    cls: Type[FilterType]  # 类属性，由子类提供

    def create_filter(self, filter: absFilterItems = None):
        if filter is None and not hasattr(self, "cls"):
            errmsg = f"Subclass {self.__class__.__name__} must define 'cls' attribute, or override create_filter method"
            raise ValueError(errmsg)
        if filter is None:
            filter = self.cls()
        filter.__dict__.update(self.request.json)
        return filter

    @response
    async def post(self):
        """
        获取分页列表
        """
        flt = self.create_filter()
        total, data = await self.actuator.query_page(flt)
        if total is None or data is None:
            return Result.fail("查询失败")
        else:
            return Page_Result.success(data, total=total)


class AbsQueryAndAddView(AbsQueryView):
    """
    基础查询和增加视图

    以前很多UI是 post 为查询，put 为增加
    做这个超类一座兼容
    """
    routePath = "/"

    poCls: Type[BasePO]

    @property
    def add_option(self) -> OperationOption:
        """
        获取添加选项
        """
        if self.poCls is None:
            raise ValueError("unoverride add_option , poCls must be defined")
        return OperationOption.create_add(self.json, poType=self.poCls, userId=self.userId)

    @response
    async def put(self):
        """
        添加
        """
        return await self.actuator.add(self.add_option)


class AbsSelectView(AuthMethodView, ABC):
    """
    基础select视图

    需要实现get_sql方法，返回sqlalchemy Select对象
    该方法用于构建select语句
    """

    routePath = "/select"

    @abstractmethod
    def get_sql(self) -> Select:
        raise NotImplementedError("get_sql method must be implemented")

    @response
    async def post(self):
        """
        获取select列表
        """
        select = self.get_sql()
        data = await self.actuator.query_all_mappings(select)
        return Result.success(data)


class AbsExistView(AuthMethodView, ABC):
    """
    PO的的某个属性值在数据库中是否存在
    """

    routePath = "/exist/<code:str>/<pk:int>"

    @property
    def column(self):
        return "*"

    @property
    @abstractmethod
    def exist_condition(self) -> Tuple[ColumnElement[bool], ...]:
        """
        存在条件
        返回一个元组，查询条件

        return (sysConfigPO.code == self.param_code, sysConfigPO.id != self.param_pk)
        """

        # return  sysConfigPO.code == self.param_code, sysConfigPO.id != self.param_pk, sysConfigPO.id != 0
        raise NotImplementedError("get_sql method must be implemented")

    def message(self, code: str, exist: bool):
        return f"编码'{code}'{'已存在' if exist else '不存在'}。"

    @property
    def param_code(self):
        return self.match_info["code"]

    @property
    def param_pk(self):
        """
        主键 当增加时 为 0
        """
        return self.match_info["pk"]

    def _response(self, code, result: bool):
        return self.response_json(
            Result.success(data=result, message=self.message(code, result))
        )

    async def get(self):
        """
        是否存在
        """
        result = await self.actuator.exist(*self.exist_condition, column=self.column)
        return self._response(self.param_code, result)


class AbsAssociationView(AuthMethodView, ABC):
    routePath = "/association/<userId:int>"

    @property
    def routePathKey(self):
        result = self.routePath.split(":")[0]
        index = result.find("<")
        return result[index + 1:]

    @property
    def routeValue(self):
        """routePath 参数里面的值"""
        return self.match_info[self.routePathKey]

    @property
    @abstractmethod
    def association_sql(self) -> Select | Tuple[Select, Select]:
        """
        subSelect = (
            Select(userGroupProjectPO.projectId, userGroupProjectPO.userGroupId)
            .filter(userGroupProjectPO.projectId == self.routeValue)
            .subquery()
        )
        select = (
            Select(
                UserGroupPO.id,
                UserGroupPO.name,
                UserGroupPO.code,
                subSelect.c.projectId.label("associatedValue"),
            )
            .outerjoin_from(
                UserGroupPO,
                subSelect,
                onclause=subSelect.c.userGroupId == UserGroupPO.id,
                full=False,
            )
            .order_by(UserGroupPO.name.asc())
        )
        """
        raise NotImplementedError("get_associationParam method must be implemented")

    @property
    def is_tree(self):
        """
        是否是树结构
        """
        if not hasattr(self, "_is_tree"):
            self._is_tree = True
        return self._is_tree

    @is_tree.setter
    def is_tree(self, value: bool):
        self._is_tree = value

    @property
    def pid_field(self):
        """
        查询树形结构使用
        父字段
        """
        if not hasattr(self, "_pid_field"):
            self._pid_field = "parentId"
        return self._pid_field

    @pid_field.setter
    def pid_field(self, value: str):
        self._pid_field = value

    @property
    def id_field(self):
        """
        查询树形结构使用
        主键字段
        """
        if not hasattr(self, "_id_field"):
            self._id_field = "id"
        return self._id_field

    @id_field.setter
    def id_field(self, value: str):
        self._id_field = value

    @property
    def root_value(self):
        """
        查询树形结构使用
        根节点值
        """
        if not hasattr(self, "_root_value"):
            self._root_value = 0
        return self._root_value

    @root_value.setter
    def root_value(self, value: int):
        self._root_value = value

    async def _query_tree(self, select: Select):
        data = await self.actuator.query_tree(
            select,
            rootValue=self.root_value,
            pid_field=self.pid_field,
            id_field=self.id_field,
        )
        return self.response_json(Result.success(data))

    async def _query_list(self,  select: Select, totalSelect: Select = None):
        total = None
        if totalSelect is not None:
            total = await self.actuator.count(totalSelect)
        data = await self.actuator.query_all_mappings(select)
        if total is None:
            return self.response_json(Result.success(data))
        return self.response_json(Page_Result.success(data, total=total))

    async def post(self):
        """
        获取关联列表
        """
        selectTotal = None
        select = self.association_sql
        if isinstance(select, Tuple) and len(select) == 2:
            selectTotal, select = select
        elif isinstance(select, Tuple) and len(select) != 2:
            raise ValueError("association_sql must be Tuple[Select,Select] or Select")
        if self.is_tree:
            return await self._query_tree(select)
        else:
            return await self._query_list(select, selectTotal)

    @property
    @abstractmethod
    def delete_sql(self) -> Delete:
        """
        删除关联语句

        param = self.get_associationParam()
        del_sql=Delete(userGroupProjectPO).filter(
            userGroupProjectPO.projectId == self.routeValue,
            userGroupProjectPO.userGroupId.in_(param.remove),
        )
        return del_sql
        """
        raise NotImplementedError("delete_sql property must be implemented")

    @abstractmethod
    async def create_association_po(
        self, session: AsyncSession, associationed_id: int, *args, **kwargs
    ) -> BasePO:
        """
        创建关联语句

        po = userGroupProjectPO()
        po.userGroupId = associationed_id
        po.projectId = self.routeValue
        return po
        """
        raise NotImplementedError("create_association_po method must be implemented")

    async def put(self):
        """
        保存关联
        {
            add:[],
            remove:[]
        }
        """
        userId = self.userId
        param = self.get_associationParam()
        sml = self.delete_sql
        return await self.save_association(
            userId, sml, self.create_association_po, param
        )


class AbsAddView(AuthMethodView, ABC):
    routePath = "/add"

    cls: Type[BasePO]  # 类属性，由子类提供

    @property
    def add_option(self) -> OperationOption:
        """
        获取添加选项
        """
        return OperationOption.create_add(self.json, poType=self.cls, userId=self.userId)

    @response
    async def post(self):
        """
        添加
        """
        return await self.actuator.operation(self.add_option) 


class AbsPkView(AuthMethodView, ABC):
    """
    可实现编辑和删除操作
    """

    cls: Type[BasePO]  # 类属性，由子类提供
    routePath = "/<pk:int>"

    @property
    def routePathKey(self):
        result = self.routePath.split(":")[0]
        index = result.find("<")
        return result[index + 1:]

    @property
    def routeValue(self):
        return self.match_info[self.routePathKey]

    @property
    def edit_option(self) -> OperationOption:
        """
        获取编辑选项
        如果需要其他参数，需要在子类中实现
        """
        if self.cls is None:
            raise ValueError("cls must be implemented")

        return OperationOption.create_edit(
            self.json, self.cls, pk=self.routeValue, userId=self.userId
        )

    @property
    def delete_option(self) -> OperationOption:
        """
        获取编辑选项
        """
        if self.cls is None:
            raise ValueError("cls must be implemented")
        return OperationOption.create_del(pk=self.routeValue, poType=self.cls)

    @response
    async def put(self):
        """
        编辑
        """
        return await self.actuator.operation(self.edit_option)

    @response
    async def delete(self):
        """
        删除
        """
        return await self.actuator.operation(self.delete_option)
