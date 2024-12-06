
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from ..base_view import AuthMethodView
from co6co.utils import find_files
from ...model.filters.file_param import FileParam
import os
import datetime


class File:
    isFile: bool
    name: str
    path: str
    right: str
    date: datetime.datetime
    size: int

    def __init__(self):
        self.isFile = None
        self.name = None
        self.path = None
        self.right = None
        self.updateTime = None
        self.size = None
        pass

    def __init__(self, root, name):
        self.name = name
        self.path = os.path.join(root, name)
        self.isFile = os.path.isfile(self.path)
        if self.isFile:
            self.size = os.path.getsize(self.path)
        self.right = None
        self.updateTime = datetime.datetime.fromtimestamp(os.path.getmtime(self.path))
        pass


class FileViews(AuthMethodView):
    async def post(self, request: Request):
        """
        列表
        """
        param = FileParam()
        param.__dict__.update(request.json)
        if param.root == None:
            param.root = "/"

        def filter(x): return param.name == None or param.name in x
        list = os.listdir(param.root)
        result = []
        for s in list:
            if filter(s):
                folder = File(param.root, s)
                result.append(folder)
        return self.response_json(Result.success({"root": param.root, "res": result}))
