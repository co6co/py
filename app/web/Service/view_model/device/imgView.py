
from sanic.response import text, file, raw, json
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select, Update, Insert
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView, BaseMethodView
from model.pos.tables import DevicePO
from view_model._filters.device import Filter
from co6co.utils import log
import ipaddress
from services.tasks.custom.devCapImg import DeviceCuptureImage
import os


class Views(BaseMethodView):
    def init(self):
        _, _, root = DeviceCuptureImage.queryConfig()
        self.root = root

    async def get(self, request: Request):
        self.init()
        items = os.listdir(self.root)
        folders = [item for item in items if os.path.isdir(os.path.join(self.root, item))]
        return self.response_json(Result.success(folders))

    async def post(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """
        name = request.json.get("date")
        self.init()
        log.warn(name)
        folder = os.path.join(self.root, name)
        # 定义常见图片文件后缀
        image_extensions = ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff']

        # 存储图片文件的列表
        image_files = []

        # 遍历当前目录下的所有文件
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in image_extensions:
                    image_files.append(file)
        return self.response_json(Result.success(image_files))


class PreView(Views):
    routePath = "/preview/<date:str>/<name:str>"

    def get(self, date: str, name: str):
        self.init()
        log.warn(date, name)
        return file(os.path.join(self.root, date, name))
