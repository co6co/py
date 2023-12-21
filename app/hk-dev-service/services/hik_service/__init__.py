# coding=utf-8

import ctypes
import os
import platform
import time
import tkinter
import json
from co6co.utils import log
from ctypes import *
import requests
import xmltodict 
from xml.dom.minidom import parseString
 
from services.hik_service.HCNetSDK import *


class HkServiceWeb:
    ip: str = None
    userName: str = None
    password: str = None

    def __init__(self, ip, userName, password) -> None:
        self.ip = ip
        self.userName = userName
        self.password = password
        pass

    def putDevice(self,dataxml:str): 
        url = f"http://{self.ip}//ISAPI/System/deviceInfo"
        response = requests.put(url, auth=(self.userName, self.password),data=dataxml)
        if response.status_code == 200:
            d=xmltodict.parse(response.content) 
            log.succ(f"更新设备信息：{d}") 
        else:
            print("Failed to start recording.")
    def getDeviceInfo(self):
        url = f"http://{self.ip}//ISAPI/System/deviceInfo"
        response = requests.get(url, auth=(self.userName, self.password))
        #response = requests.put(url, auth=(self.userName, self.password))
        if response.status_code == 200:
            d=xmltodict.parse(response.content)
            log.warn(d)
            d.update({"DeviceInfo":{"deviceName":"IP CAMERA"}})
            data =xmltodict.unparse(d)  
            log.warn(d)
            log.warn(data)
            self.putDevice(data)
        else:
            print("Failed to start recording.")


class HkService:
    # 系统环境标识
    windiws_flage: bool = True
    userId: int = None
    sdk: object = None

    def __init__(self) -> None:
        # 获取系统平台
        self.__getPlatform()
        # 加载库,先加载依赖库
        if self.windiws_flage:
            os.chdir(r'./lib/win')
            self.sdk = ctypes.CDLL(r'./HCNetSDK.dll')
        else:
            os.chdir(r'./lib/linux')
            self.sdk = cdll.LoadLibrary(r'./libhcnetsdk.so')

        self.SetSDKInitCfg()  # 设置组件库和SSL库加载路径

        # 初始化
        self.sdk.NET_DVR_Init()
        # 启用SDK写日志
        # self.sdk.NET_DVR_SetLogToFile( 3, bytes('./SdkLog_Python/', encoding="utf-8"), True)
        self.sdk.NET_DVR_SetLogToFile(
            3, bytes('C:\\SdkLog\\', encoding="utf-8"), True)

        '''
        # 注册登录设备
        userId = self.Login()
        if userId < 0:
            print('登录失败，退出')
        else:
            self.setShowString(userId)  # 透传方式设置通道字符叠加参数
            self.getShowString(userId)  # 透传方式获取通道字符叠加参数
        '''

    def __getPlatform(self):
        """
        获取当前系统环境
        """
        sysstr = platform.system()
        print('' + sysstr)
        if sysstr != "Windows":
            # global WINDOWS_FLAG
            self.windiws_flage = False
        else:
            self.windiws_flage = True

    def SetSDKInitCfg(self):
        # 设置SDK初始化依赖库路径
        # 设置HCNetSDKCom组件库和SSL库加载路径
        # print(os.getcwd())
        if self.windiws_flage:
            strPath = os.getcwd().encode('gbk')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.sdk.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.sdk.NET_DVR_SetSDKInitCfg(
                3, create_string_buffer(strPath + b'\libcrypto-1_1-x64.dll'))
            self.sdk.NET_DVR_SetSDKInitCfg(
                4, create_string_buffer(strPath + b'\libssl-1_1-x64.dll'))
        else:
            strPath = os.getcwd().encode('utf-8')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.sdk.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.sdk.NET_DVR_SetSDKInitCfg(
                3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
            self.sdk.NET_DVR_SetSDKInitCfg(
                4, create_string_buffer(strPath + b'/libssl.so.1.1'))

    def Login(self, ip: str, user: str, password: str):
        # 设备登录信息
        struLoginInfo = NET_DVR_USER_LOGIN_INFO()
        struLoginInfo.bUseAsynLogin = 0  # 同步登录方式
        struLoginInfo.sDeviceAddress = bytes(ip, "ascii")  # 设备IP地址
        struLoginInfo.wPort = 8000  # 设备服务端口
        struLoginInfo.sUserName = bytes(user, "ascii")  # 设备登录用户名
        struLoginInfo.sPassword = bytes(password, "ascii")  # 设备登录密码

        # 设备信息, 输出参数
        struDeviceInfoV40 = NET_DVR_DEVICEINFO_V40()

        # 登录设备
        self.userId = self.sdk.NET_DVR_Login_V40(
            byref(struLoginInfo), byref(struDeviceInfoV40))
        if self.userId < 0:
            print('登录失败, 错误码: %d' % self.sdk.NET_DVR_GetLastError())
        else:
            print('登录成功，设备序列号：%s' % str(
                struDeviceInfoV40.struDeviceV30.sSerialNumber, encoding="utf8"))

    def Close(self):
        if self.userId!=None and self.userId >= 0:
            # 注销用户
            self.sdk.NET_DVR_Logout(self.userId)
            # 释放SDK资源
            # log.warn(dir(self.sdk))
            # self.sdk.NET_DVE_Cleanup() # 没有该方法

    def create_input(self, url: str, queryData: str = None) -> NET_DVR_XML_CONFIG_INPUT:
        xmlInput = NET_DVR_XML_CONFIG_INPUT()
        xmlInput.dwSize = sizeof(xmlInput)
        url = create_string_buffer(bytes(url, encoding="ascii"))
        xmlInput.lpRequestUrl = addressof(url)
        xmlInput.dwRequestUrlLen = len(url)
        xmlInput.lpInBuffer = None
        xmlInput.dwInBufferSize = 0
        if queryData != None:
            str_bytes = bytes(queryData, encoding="ascii")
            xmlInput.lpInBuffer = cast(str_bytes, c_void_p)
            xmlInput.dwInBufferSize = len(str_bytes)
        xmlInput.dwRecvTimeOut = 5000
        xmlInput.byForceEncrpt = 0
        return xmlInput

    def create_output(self) -> NET_DVR_XML_CONFIG_OUTPUT:
        xmlOutput = NET_DVR_XML_CONFIG_OUTPUT()
        xmlOutput.dwSize = sizeof(xmlOutput)
        xmlOutput.dwOutBufferSize = 8 * 1024
        xmlOutput.dwStatusSize = 1024
        M1 = 8 * 1024
        buff1 = (c_ubyte * M1)()
        M2 = 1024
        buff2 = (c_ubyte * M2)()

        xmlOutput.lpOutBuffer = addressof(buff1)
        xmlOutput.lpStatusBuffer = addressof(buff2)
        return xmlOutput

    def request2(self, xmlInput: str, xmlOutput: str = None):
        reValue = self.sdk.NET_DVR_STDXMLConfig(
            self.userId, byref(xmlInput), byref(xmlOutput))
        result: str = None
        if reValue == 1:
            Bbytes_Status = string_at(
                xmlOutput.lpStatusBuffer, xmlOutput.dwStatusSize)
            result = str(Bbytes_Status, 'UTF-8')
        else:
            # https://open.hikvision.com/hardware/definitions/NET_DVR_GetLastError.html
            log.err(f" NET_DVR_GetLastError:{ self.sdk.NET_DVR_GetLastError()},登录句柄:{self.userId}")

        return result

    def request(self, url: str, inputData: str = None):
        xmlInput = self.create_input(url, inputData)
        xmlOutput = self. create_output()
        reValue = self.sdk.NET_DVR_STDXMLConfig(
            self.userId, byref(xmlInput), byref(xmlOutput))
        result: str = None
        if reValue == 1:
            Bbytes_Status = string_at(
                xmlOutput.lpStatusBuffer, xmlOutput.dwStatusSize)
            result = str(Bbytes_Status, 'UTF-8')
        else:
            # https://open.hikvision.com/hardware/definitions/NET_DVR_GetLastError.html
            print(f"{url},NET_DVR_GetLastError:{ self.sdk.NET_DVR_GetLastError()},登录句柄:{self.userId}")

        return result

    def setShowString(self):
        url = "PUT /ISAPI/System/Video/inputs/channels/1/overlays/text/1"
        querydata = ('<TextOverlay xmlns=\"http://www.hikvision.com/ver20/XMLSchema\" version=\"2.0\">'
                     '<id>1</id>'
                     '<enabled>true</enabled>'
                     '<positionX>100</positionX>'
                     '<positionY>200</positionY>'
                     '<displayText>1234567测试abc</displayText></TextOverlay>')

        self .request(url, querydata)

    def getShowString(self):
        url = 'GET /ISAPI/System/Video/inputs/channels/1/overlays/text/1'
        data = self .request(url)
        print(f"请求{url},结果：{data}")

    def getDeviceInfo(self):
        url = 'GET /ISAPI/System/deviceInfo'
        data = self.request(url)
        print(f"请求{url},结果：{data}")

    # 区域局部聚焦和局部曝光功能：矩形区域座标左上角和右下角（startX,startY,endX,endY）
    # flag=1局部聚焦功能，flag!=1局部曝光功能
    def RegionalCorrection(self, startX, startY, endX, endY, flag=1):
        # #定义传输内容
        if (flag == 1):
            choise = "regionalFocus"
        else:
            choise = "regionalExposure"
        inUrl = "PUT /ISAPI/Image/channels/1/" + choise
        inPutBuffer = "<" + choise + "><StartPoint><positionX>" + str(startX) + "</positionX><positionY>" + str(
            startY) + "</positionY></StartPoint><EndPoint><positionX>" + str(endX) + "</positionX><positionY>" + str(endY) + "</positionY></EndPoint></" + choise + ">"
        self.request(inUrl, inPutBuffer)

        input = self.create_input(inUrl, inPutBuffer)

        szUrl = (ctypes.c_char * 256)()
        struInput = NET_DVR_XML_CONFIG_INPUT()
        struOuput = NET_DVR_XML_CONFIG_OUTPUT()
        struInput.dwSize = ctypes.sizeof(struInput)
        struOuput.dwSize = ctypes.sizeof(struOuput)
        dwBufferLen = 1024 * 1024
        pBuffer = (ctypes.c_char * dwBufferLen)()

    # _____________________________________________put________________________________________________________
        csCommand = bytes(inUrl, "ascii")
        ctypes.memmove(szUrl, csCommand, len(csCommand))
        struInput.lpRequestUrl = ctypes.cast(szUrl, ctypes.c_void_p)
        struInput.dwRequestUrlLen = len(szUrl)

        m_csInputParam = bytes(inPutBuffer, "ascii")
        dwInBufferLen = 1024 * 1024
        pInBuffer = (ctypes.c_byte * dwInBufferLen)()
        ctypes.memmove(pInBuffer, m_csInputParam, len(m_csInputParam))
        struInput.lpInBuffer = ctypes.cast(pInBuffer, ctypes.c_void_p)
        struInput.dwInBufferSize = len(m_csInputParam)

        struOuput.lpStatusBuffer = ctypes.cast(pBuffer, ctypes.c_void_p)
        struOuput.dwStatusSize = dwBufferLen

        self.request2(input, struOuput)
        if (self.sdk.NET_DVR_STDXMLConfig(self.userId, ctypes.byref(struInput), ctypes.byref(struOuput))):
            error_info = self.sdk.NET_DVR_GetLastError()
            print("上传成功：" + str(error_info))
        else:
            error_info = self.sdk.NET_DVR_GetLastError()
            print("上传失败：错误号为" + str(error_info))
