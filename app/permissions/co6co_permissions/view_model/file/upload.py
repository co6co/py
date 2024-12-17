
from sanic.response import text
from sanic import Request
from sanic.response import file, file_stream, json, raw
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from ..base_view import AuthMethodView
from ...model.filters.file_param import FileParam
import os
import datetime
from co6co.utils import log
import tempfile
import shutil


class UploadView(AuthMethodView):
    routePath = "/upload"

    async def get(self, request: Request):
        file_name = request.args.get('fileName')
        args = self.usable_args(request)
        filePath = args.get("path")

        if not filePath:
            return self.response_json(Result.fail(message="缺少文件文件路径"))

        uploaded_chunks = []
        for i in range(1, 1000):  # 假设最多有 1000 个块
            temp_file = tempfile.NamedTemporaryFile(delete=False)

            temp_file_path = os.path.join(TEMP_FOLDER, f'{file_name}_part{i}')
            if os.path.exists(temp_file_path):
                uploaded_chunks.append(i)
            else:
                break
        return self.response_json(Result.success({'uploadedChunks': uploaded_chunks}))

    async def put(self, request: Request):
        """
        上传 chunk
        """
        file = request.files.get('file')
        index = int(request.form.get('index'))
        total_chunks = int(request.form.get('totalChunks'))
        file_name = request.form.get('fileName')

        if not file or not file_name:
            return self.response_json(Result.fail(message="缺少文件名"))

        # 保存文件块到临时目录
        temp_file_path = os.path.join(TEMP_FOLDER, f'{file_name}_part{index}')
        file.save(temp_file_path)

        # 检查是否所有块都已上传
        if index == total_chunks:
            merge_chunks(file_name, total_chunks)

        return self.response_json(Result.success(message="文件块上传成功"))


def merge_chunks(self, file_name, total_chunks):
    """
    合并文件块
    """
    output_file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with open(output_file_path, 'wb') as output_file:
        for i in range(1, total_chunks + 1):
            temp_file_path = os.path.join(TEMP_FOLDER, f'{file_name}_part{i}')
            with open(temp_file_path, 'rb') as temp_file:
                output_file.write(temp_file.read())
            os.remove(temp_file_path)  # 删除临时文件

    print(f'文件 {file_name} 合并完成')
