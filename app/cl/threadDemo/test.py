
from co6co.utils import log, try_except
import os
from model.pos.tables import DevicePO
import cv2

from co6co_db_ext .session import dbBll
from co6co_db_ext.db_utils import QueryListCallable, db_tools
from co6co.utils import log

from sqlalchemy.sql import Select, Update
from typing import List, Tuple, TypedDict
from co6co.utils import getDateFolder
from pathlib import Path 
from co6co.task.pools import limitThreadPoolExecutor 
from concurrent.futures import Future
from co6co.task.utils import Timer
import time
from model.enum import DeviceCategory, DeviceCheckState, DeviceVender
from co6co.utils import network
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession 
class TableEntry(TypedDict):
    name: str
    code: str
    ip: str
    vender: str
    userName: str
    passwd: str
    category: int


def mask_rtsp_credentials(rtsp_url: str):
    # 查找"//"的位置
    double_slash_index = rtsp_url.find("//")
    if double_slash_index == -1:
        return rtsp_url  # 不包含"//"，直接返回原URL

    # 从"//"之后查找"@"的位置
    at_index = rtsp_url.rfind("@", double_slash_index + 2)
    if at_index == -1:
        return rtsp_url  # 不包含"@"，直接返回原URL

    # 替换"//"和"@"之间的内容为"*"

    userPwd = rtsp_url[double_slash_index + 2:at_index]
    at_index2 = userPwd.find(":")
    mask = ""
    if at_index2 > -1:
        user = userPwd[:at_index2]
        pwd = userPwd[at_index2+1:]
        mask = "*"*len(user) + ":" + "*"*len(pwd)+"@"
    else:
        mask = userPwd+"@"
    return rtsp_url[:double_slash_index + 2] + mask + rtsp_url[at_index+1:]


class DeviceCuptureImage( ):
    name = "抓图设备流图片"
    code = "CAPTURE_DEV_IMAGE"
    root_cache_key = "__cap_img_key__"

     
    def createDbBll(self):
        db_settings={"DB_HOST":"localhost","DB_PORT":3306,"DB_USER":"root","DB_PASSWORD":"mysql123456","DB_NAME":"dggy_db","echo":True}
        return dbBll( db_settings=db_settings  )
    def __init__(self): 
        super().__init__() 
        self._deviceCount = 0
        self.isQuit=False

    def getSubPath(self, filePath: str):
        """
        获取URL表示的路径 /upload/2025-01-01/12.jpeg
        """
        p_path = Path(filePath)
        root_path = Path(self.root)

        # 获取相对路径
        relative_path = p_path.relative_to(root_path)
        # 转换为以斜杠开头的字符串
        result = "/" + str(relative_path).replace("\\", "/")
        return result

    async def queryTable(self, session: AsyncSession) -> List[TableEntry]:
        try:
            call = QueryListCallable(session)
            select = (
                Select(DevicePO.name, DevicePO.code, DevicePO.ip, DevicePO. vender, DevicePO.userName, DevicePO.passwd, DevicePO.category)
                .filter(DevicePO.state == 1)
            )  
            return await call(select, isPO=False)

        except Exception as e:
            log.err("执行 ERROR", e)
            return []

    async def update_state_to_db_task(self, session: AsyncSession, ip: str, checkState: DeviceCheckState, stateDesc: str = None, imgPath: str = None):
        try: 
            if not stateDesc:
                stateDesc = checkState.label 
            async with session, session.begin(): 
                result = await db_tools.execSQL(
                    session,
                    Update(DevicePO).where(DevicePO.ip == ip).values(checkState=checkState.val, checkDesc=stateDesc, checkImgPath=imgPath, checkTime=datetime.now())
                ) 
                log.info(f"更新设备{ip}状态为{checkState.val}- {stateDesc},影响行数:{result}") 
        except Exception as e:
            log.err(f"更新状态异常：{ip}",e)

    def update_state_to_db(self, ip: str, checkState: DeviceCheckState, stateDesc: str = None, imgPath: str = None):
        try:
            bll =self.createDbBll()
            bll.run(self.update_state_to_db_task, bll.session, ip, checkState, stateDesc, imgPath) 
        except Exception as e: 
            log.err(f"更新设备{ip}状态失败", e)
        finally: 
            bll.close()
            log.succ(ip,"update_state_to_db_task关bll连接",str(bll.closed))

    @staticmethod
    def queryConfig(bll: dbBll = None): 
        #deviceConfig: dict | None = bll.run(bll.query_config_value, "device_config", True)
        deviceConfig=None
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
        bll = self.createDbBll()
        log.warn("查询配置...")
        userName, password, root, date = ('admin','bwk68240175',"D:\\temp","%Y-%m-%d-%H")#DeviceCuptureImage.queryConfig(bll)

        log.warn("查询设备...")
        result = bll.run(self.queryTable, bll.session) 
        log.succ("查询设备..") 
        bll.close() 
        log.warn("查询设备.")
        return result, userName, password, root, date

    def check_network(self, ip: str):
        """
        检测网络是否通畅
        """ 
        pingResult = network.ping_host(ip)
        if not pingResult:
            log.warn(f"{ip} 网络不通")
        return pingResult 
        
    def check_all(
        self,
        ip: str,
        port: int = 554,
        video_path: str = None,
        output_image_path: str = None
    ) -> None:
        try:
            # 先检查网络连接
            if not ip:
                return  # IP为空时直接返回，无需后续检查
            is_network_ok = self.check_network(ip) 
            if not is_network_ok:
                self.update_state_to_db(ip, DeviceCheckState.networkError)
            self.update_state_to_db(ip, DeviceCheckState.normal) 
        except Exception as e:
            log.err(f"检查设备{ip}端口失败", e) 

     
    def result(self, f: Future):
        try:
            if f.exception():
                log.err(f"执行{f.ip} ERROR", f.exception())
                print(f"发生错误: {f.exception()}")
            else:
                result=f.result()
                log.succ(f"{f.ip}执行结果：{result}")
        except Exception as e: 
            log.warn("执行result ERROR", str(e), "Future canceled->", f.cancelled())
        finally:
            self._deviceCount -= 1
            log.warn(f"剩余设备数量{self._deviceCount}个,currentIP:{ip}")

    @try_except
    def getRtspAddress(self, category: DeviceCategory, ip, userName, pwd, deviceName, vender: str):
        # 用户名为 _ 代码系统用户发现设备用户名不正确，不进行检测
        if userName == "_":
            return []
        path = Path(self.root) / self.subFolder
        path.exists() or os.makedirs(path)
        output_image_path = path/f"{deviceName}_{ip}.jpg"
        v: DeviceVender = DeviceVender.key2enum(vender.upper()) if vender else DeviceVender.Hikvision
        result = []
        video_path = None
        if v.val == DeviceVender.Hikvision.val:
           # 一体机有两个通道
            if category == DeviceCategory.ParkAndPass:
                video_path = f"rtsp://{userName}:{pwd}@{ip}:554/Streaming/Channels/1"
                video_path_2 = f"rtsp://{userName}:{pwd}@{ip}:554/Streaming/Channels/2"
                output_image_path_2 = path/f"{deviceName}_{ip}_2.jpg"
                result.append((video_path, output_image_path), (video_path_2, output_image_path_2))
            else:
                video_path = f"rtsp://{userName}:{pwd}@{ip}:554/Streaming/Channels/1"

        elif v.val == DeviceVender.Uniview.val:
            # /media/video1     主码流
            # /media/video2     辅码流
            video_path = f"rtsp://{userName}:{pwd}@{ip}:554/media/video1"
            pass
        elif v.val == DeviceVender.Dahua.val:
            # rtsp://192.168.2.235:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif
            # <channel>是摄像头通道编号，这里是 “1”，表示第一通道。
            # <subtype>表示子流类型，“0” 表示主码流，“1” 表示子码流，此处 “0” 代表主码流。
            # video_path = f"rtsp://{userName}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0"
            video_path = f"rtsp://{userName}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0"
            pass

        elif v.val == DeviceVender.TPLink.val:
            pass
        if video_path:
            result.append((video_path, output_image_path))
        return result

     
    @try_except
    def main_ext(self):
        deviceList, userName, pwd, root, date = self.queryAllDevice()
        self.subFolder = date
        self.root = root
        #DeviceCuptureImage.setCapImgRootCache(root)
        # print(deviceList, userName, pwd)
        timeer = Timer(f"{self.name}:{time.time()}", showMsg=False)
        log.warn(f"{self.name}开始...", len(deviceList))
        timeer.start()
        # 遍历设备列表
        deviceCount = 0
        deviceLen = len(deviceList)
        self._deviceCount = deviceLen

        def bck(f: Future, ip: str, video_path: str = None, output_image_path: str = None):
            f.path = video_path
            f.ip = ip
            f.savePath = output_image_path 
            f.add_done_callback(self.result)
        print("开始检测...") 
        with limitThreadPoolExecutor(max_workers=4, thread_name_prefix="capture_pj") as executor: 
            for device in deviceList:
                try:
                    deviceCount += 1
                    if self.isQuit and not executor._shutdown:
                        log.warn("关闭线程池,不再接收新任务...,等等正在执行的任务完成...")
                        executor.shutdown(False, cancel_futures=True)
                        break

                    newUser = device['userName'] or userName
                    newPwd = device['passwd'] or pwd
                    category = device['category'] or 0
                    ip = device['ip']
                    deviceName = device['name']
                    vender = device['vender']
                    f: Future = None
                    deviceCategory: DeviceCategory = DeviceCategory.val2enum(category)
                    log.succ(f"{deviceCount}/{deviceLen}->{deviceCategory},{deviceCategory.val},{deviceCategory.hasVideo()}")
                    #if deviceCategory.hasVideo():
                    #    rtspList = self.getRtspAddress(deviceCategory, ip, newUser, newPwd, deviceName, vender)
                    #    if not rtspList or len(rtspList) == 0:
                    #        log.warn(f"{vender},{deviceName, }{ip}没有视频地址！")
                    #        f = executor.submit(self.check_all, ip, None) 
                    #        bck(f, ip)
                    #        continue
                    #    for video_path, output_image_path in rtspList:
                    #        f = executor.submit(self.check_all,   ip, video_path=video_path,  output_image_path=output_image_path) 
                    #        bck(f, ip, video_path=video_path,  output_image_path=output_image_path)
                    #else:
                    #    f = executor.submit(self.check_all, ip, None)
                    #    bck(f, ip) 
                    f = executor.submit(self.check_all, ip, None) 
                    #f = executor.submit(network.ping_host, ip ) 
                    bck(f,ip) 
                except Exception as e:
                    log.err("ERR", e)

        beforeShow = True
        if executor._shutdown:
            log.succ("等待正在运行的线程退出..")
        else:
            log.succ("等待所有任务结束....")
            beforeShow = False
            executor.shutdown(True)

        timeer.stop()
        log.succ(f"{timeer.activity_name}完成,提前结束—>{beforeShow},耗时->{timeer.elapsed}秒", )

    def main(self):
        self.main_ext()
        log.succ("devCapImg Exit")
    def __del__(self) -> None:
        log.info("devCapImg quited")
    


if __name__=="__main__":
    d=DeviceCuptureImage()
    d.main()