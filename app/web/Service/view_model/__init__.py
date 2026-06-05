# -*- coding: utf-8 -*-

# 导入基础
from sanic import response, json
from sanic.views import HTTPMethodView
from typing import Callable, Any
from sanic.response import text, file, raw, json
from sanic import Request
from co6co.data.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from co6co_permissions.view_model.base_view import AuthMethodView

from co6co.data.excell import AbsImport
from typing import Literal


class ImportView(AuthMethodView, AbsImport):
    routePath = "/import"

    def __init__(self, sheet_name, templateFileName):
        AbsImport.__init__(self, sheet_name, templateFileName=templateFileName)

    def columns(self):
        columns = []
        raise OverflowError("子类必须实现columns方法")

    async def head(self, request: Request):
        """
        文件
        """
        return await self.response_head(self.template_length, self.templateFileName)

    async def _get_before(self, request: Request):
        """
        获取前处理

        需要子类实现 
        """
        return None

    async def get(self, request: Request):
        """
        获取模板文件
        """
        try:
            data = await self._get_before(request)

            binary_data = self.template(data)
            headers = {
                "Content-Disposition": f'attachment; filename="{self.templateFileName}"'
            }
            # return await file(template_path, filename='template.xlsx')
            return raw(binary_data, headers=headers)
        except FileNotFoundError:
            return self.response_json(Result.fail(message="Template file not found"))
        except Exception as e:
            return self.response_json(Result.fail(message=f"Err:{e}"))

    async def handlerBefore(self, item, **kwargs):
        return item

    async def handler(self, rawData: list[list | dict], item,  dbSession: AsyncSession, **kwargs):
        raise OverflowError("子类必须实现handler方法")

    async def handlerFinshed(self, dbSession: AsyncSession, **kwargs):
        return await dbSession.commit()

    async def handlerError(self, dbSession: AsyncSession, **kwargs):
        return await dbSession.rollback()

    async def _post(self, request: Request, orient: Literal['dict', 'list', "series", 'index', 'records', 'split'] = "records"):
        """
        records   ->   pip install xlrd
        """
        try:
            # 检查请求中是否包含文件
            if 'file' not in request.files:
                return self.response_json(Result.fail(message="No file part"))
            file = request.files.get('file')
            filename: str = file.name
            # 检查文件是否为 Excel 格式
            if not self.fileCheck(filename):
                return self.response_json(Result.fail(message="Invalid file format"))
            dbSession = self.get_db_session(request)
            insertNum, uploadNum, errorNUm, msg = await self.importData(file.body, orient=orient, request=request, dbSession=dbSession)
            if insertNum+uploadNum > 0:
                return self.response_json(Result.success(message=f"导入成功，新增{insertNum}条，更新{uploadNum}条，错误{errorNUm}条,{msg}"))
            else:
                return self.response_json(Result.fail(message=f"导入数据为0,可能执行失败失败，{msg}"))
        except Exception as e:
            return self.response_json(Result.fail(message=f"导入出现错误，{e}"))

from co6co.utils import debug
class demoView2(HTTPMethodView):
    routePath = "/demo2"
    def __init__(self, *class_args: any, **class_kwargs: any) -> None:
        debug()
        print("demoView2 init",class_args, class_kwargs)
        super().__init__(*class_args, **class_kwargs)
  

    async def get(self, request: Request, *args, **kwargs):
        response = json({"message": "Hello, World!"})
        return response

    async def post(self, request: Request, *args, **kwargs):
        response = json({"message": "Hello, World! post"})
        return response


class demoView():
    routePath = "/demo"
    decorators: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = []

    def __init__(self, request: Request, *args, **kwargs):
        print("demoView init",args, kwargs)
        self.request = request
        pass

    @classmethod
    def as_view(cls, *class_args: any, **class_kwargs: any):
        def view(*args, **kwargs):
            self: demoView = view.view_class(*args, **kwargs)
            return self()

        if cls.decorators:
            view.__module__ = cls.__module__
            for decorator in cls.decorators:
                view = decorator(view)

        view.view_class = cls  # type: ignore
        view.__doc__ = cls.__doc__
        view.__module__ = cls.__module__
        view.__name__ = cls.__name__
        return view

    async def get(self):
        response = json({"message": "Hello, World!"})
        return response

    async def post(self):
        response = json({"message": "Hello, World! post"})
        return response

    async def __call__(self) -> response.BaseHTTPResponse:
        method = self.request.method
        handler = getattr(self, method.lower(), None)
        if handler is None:
            raise NotImplementedError(f"Method {method} not implemented")
        return await handler()

    def __await__(self):
        return self.__call__().__await__()
