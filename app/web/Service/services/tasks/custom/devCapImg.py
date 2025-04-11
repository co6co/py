from .base import ICustomTask
from services.cfService import CfService
from sanic import Sanic
from co6co.utils import log
import asyncio
from co6co.utils import try_except
import os
from model.pos.tables import DevicePO
import cv2
from co6co_permissions.services.bllConfig import config_bll, BaseBll
from co6co_db_ext.db_utils import QueryListCallable
from co6co.utils import log
from co6co_permissions.model.enum import dict_state
from sqlalchemy.sql import Select, Update
from typing import List, Tuple, TypedDict
from co6co.utils import getDateFolder
from pathlib import Path
from co6co_sanic_ext import sanics
from urllib.parse import quote
from co6co.task.pools import limitThreadPoolExecutor
from concurrent.futures import Future


class TableEntry(TypedDict):
    name: str
    code: str
    ip: str
    userName: str
    passwd: str


class DeviceCuptureImage(ICustomTask):
    name = "抓图设备流图片"
    code = "CAPTURE_DEV_IMAGE"

    def __init__(self, worker: sanics.Worker = None):
        super().__init__(worker)

    async def queryTable(self, bll: BaseBll) -> List[TableEntry]:
        try:
            call = QueryListCallable(bll.session)
            select = (
                Select(DevicePO.name, DevicePO.code, DevicePO.ip, DevicePO.userName, DevicePO.passwd)
                .filter(DevicePO.state == dict_state.enabled.val)
            )
            return await call(select, isPO=False)

        except Exception as e:
            log.err("执行 ERROR", e)
            return []

    @staticmethod
    def queryConfig(bll: config_bll = None):
        bll = bll or config_bll()
        deviceConfig: dict | None = bll.run(bll.query_config_value, "device_config", True)
        userName = "admin"
        password = "password"
        root = "D:\\temp"
        quote = False
        defaultDate = "%Y-%m-%d-%H"
        date = defaultDate
        if deviceConfig:
            userName = deviceConfig.get("userName", userName)
            password = deviceConfig.get("password", password)
            root = deviceConfig.get("root", root)
            quote = deviceConfig.get("quote", quote)
            date = deviceConfig.get("date", date)
        else:
            log.warn("未找到设备配置,需配置 device_config,{userName:"",password:"",root:""}")
        if quote:
            password = quote(password)
        try:
            dateFolder = getDateFolder(date)
        except Exception as e:
            log.warn(f"日期格式'{date}'错误,请检查配置")
            dateFolder = getDateFolder(defaultDate)
        return userName, password, root, dateFolder

    def queryAllDevice(self) -> Tuple[List[TableEntry], str, str, str]:
        bll = config_bll()
        userName, password, root, date = DeviceCuptureImage.queryConfig(bll)
        return bll.run(self.queryTable, bll), userName, password, root, date

    def capture_dev_image(self, video_path, output_image_path: str):
        """
        从视频文件中捕获一帧并保存为图片。
        :param video_path: 视频文件路径。
        :param output_image_path: 输出图片文件路径（不能包含中文路径）。
        """
        # 视频文件路径
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log.warn(f"无法打开视频文件'{video_path}'!")
                return
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 将中文路径转换为字节类型
                # result = cv2.imwrite(output_image_path, frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                with open(output_image_path, 'wb') as f:
                    f.write(buffer)
                log.info("图片已保存到:", output_image_path)
            else:
                log.warn("无法读取视频帧！", video_path)

            # 释放资源
            cap.release()
        except Exception as e:
            log.err(f"打开{video_path}出现ERROR!!", e)

    def result(self, f: Future):
        if f.exception():
            log.err(f"执行{f.path} ERROR", f.exception())
            print(f"发生错误: {f.exception()}")

    @try_except
    def main(self):
        deviceList, userName, pwd, root, date = self.queryAllDevice()
        from co6co.task.utils import Timer
        import time
        # print(deviceList, userName, pwd)
        ns = time.time()
        timeer = Timer(f"抓图设备流图片:{ns}")
        timeer.start()
        pool = limitThreadPoolExecutor(max_workers=4, thread_name_prefix="capture_pj")

        # 遍历设备列表
        for device in deviceList:
            if self.worker and self.worker.isQuit:
                pool.shutdown(False, cancel_futures="工作线程退出")
                break
            newUser = device['userName'] or userName
            newPwd = device['passwd'] or pwd
            video_path = f"rtsp://{newUser}:{newPwd}@{device['ip']}:554/Streaming/Channels/1"
            # log.warn("视频地址：", video_path)
            path = Path(root) / date
            path.exists() or os.makedirs(path)
            output_image_path = path/f"{device['name']}_{device['ip']}.jpg"
            # print(video_path, output_image_path)
            f = pool.submit(self.capture_dev_image, video_path, output_image_path)
            f.path = video_path
            f.savePath = output_image_path
            f.add_done_callback(self.result)
            # self.capture_dev_image(video_path, output_image_path)

        timeer.stop()
        log.warn(f"{ns}抓图设备流图片完成,耗时：", timeer.time)
