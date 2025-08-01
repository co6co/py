
import mimetypes
from pathlib import Path
from sanic.response import text, file, raw, json
from sanic.exceptions import NotFound
from sanic import Request
from co6co_sanic_ext.model.res.result import Result, Page_Result
from co6co_permissions.view_model.base_view import AuthMethodView, BaseMethodView
from co6co.utils import log
from services.tasks.devCapImg import DeviceCuptureImage
import os
import shutil
from co6co_db_ext.db_filter import absFilterItems
from urllib.parse import unquote
from model.pos.tables import DevicePO


class FileFilter(absFilterItems):
    """
    不需要 DevicePO 的字段
    """
    date: str = None
    name: str = None

    def __init__(self, date: str = None):
        self.date = date
        self.name = None
        super().__init__(DevicePO)
        pass

    def filter(self) -> list:
        """
        过滤条件
        """
        filters_arr = []
        if self.checkFieldValue(self.date):
            filters_arr.append(DevicePO.name.like(f"%{self.date}%"))

    def getDefaultOrderBy(self):
        """
        默认排序
        """
        return (DevicePO.name.desc(),)


class Views(BaseMethodView):
    def init(self):
        _, _, root, _ = DeviceCuptureImage.queryConfig()
        self.root = root

    async def get(self, request: Request):
        self.init()
        items = os.listdir(self.root)
        folders = [item for item in items if os.path.isdir(os.path.join(self.root, item))]
        # 为每个项目获取完整路径
        full_paths = [os.path.join(self.root, item) for item in folders]

        # 按修改时间排序
        sorted_items = sorted(full_paths, key=lambda x: os.path.getmtime(x), reverse=True)
        sorted_items = [os.path.basename(item) for item in sorted_items]
        return self.response_json(Result.success(sorted_items))

    @staticmethod
    def paginated_os_walk(folder, page_size, page_number, image_extensions: list = None, allPath=False, name: str = None):
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        counter = 0
        for root, dirs, files in os.walk(folder):
            if name:
                files = [file for file in files if name in file]
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if image_extensions is None or file_extension in image_extensions:
                    if start_index <= counter < end_index:
                        if allPath:
                            yield os.path.join(root, file), len(files)
                        else:
                            yield file, len(files)
                else:
                    continue
                counter += 1

    async def post(self, request: Request):
        """
        树形选择下拉框数据
        selectTree :  el-Tree
        """

        filter = FileFilter()
        filter.__dict__.update(request.json)
        self.init()
        folder = os.path.join(self.root, filter.date)
        # 定义常见图片文件后缀
        image_extensions = ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff']

        # 存储图片文件的列表
        image_files = []
        imagesgen = Views.paginated_os_walk(folder,  filter.pageSize, filter.pageIndex, image_extensions, name=filter.name)
        total = 0
        for image, count in imagesgen:
            total = count
            image_files.append(image)
        # 遍历当前目录下的所有文件
        # for root, dirs, files in os.walk(folder):
        #    for file in files:
        #        file_extension = os.path.splitext(file)[1].lower()
        #        if file_extension in image_extensions:
        #            image_files.append(file)
        return self.response_json(Page_Result.success(image_files, total=total))


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


class DeviceCheckImage(AuthMethodView):
    routePath = "/imgpreview"

    async def get(self, request: Request):
        """
        获取图片
        """
        # 获取请求参数
        dic = self.usable_args(request)
        img_path: str = dic.get("path", None)
        # 检查路径参数是否存在
        if not img_path:
            return raw("图片路径参数 'path' 缺失", 500)

        # 处理路径，移除开头的斜杠（如果存在）
        if img_path.startswith('/'):
            img_path = img_path[1:]

        # 获取图片根目录
        root = DeviceCuptureImage.getCapImgRootCache()
        if not root:
            return raw("根路径未找到，请稍后重试", 500)

        # 构建完整路径
        root_path = Path(root)
        full_path = root_path / img_path

        # 检查文件是否存在且是文件
        if not full_path.exists():
            return raw(f"图片不存在: {full_path}", 404)
        if not full_path.is_file():
            return raw(f"路径 {full_path}不是文件", 404)

        # 自动检测MIME类型
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = "application/octet-stream"  # 默认二进制类型

        # 返回文件响应
        return await file(
            full_path,
            mime_type=mime_type
        )


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
