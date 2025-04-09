
from sanic.response import text, file, raw, json
from sanic.exceptions import NotFound
from sanic import Request
from co6co_sanic_ext.model.res.result import Result
from co6co_permissions.view_model.base_view import AuthMethodView, BaseMethodView
from co6co.utils import log
from services.tasks.custom.devCapImg import DeviceCuptureImage
import os
import shutil

from urllib.parse import unquote


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

    def get(self, request: Request, date: str, name: str):
        self.init()
        self.parser_multipart_body
        date = unquote(date)
        name = unquote(name)
        filePath = os.path.join(self.root, date, name)
        log.warn(filePath)
        if os.path.exists(filePath):
            return file(filePath)
        else:
            return NotFound("图片不存在")


class DeleteFolderViews(Views):
    routePath = "/<date:str>"

    async def delete(self, request: Request, date: str):
        """
        删除
        """
        self.init()
        folder = os.path.join(self.root, date)
        if os.path.exists(folder):
            shutil.rmtree(folder)
            return self.response_json(Result.success())
        else:
            return self.response_json(Result.fail("文件夹不存在"))
