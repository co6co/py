from co6co_sanic_ext.model.res.result import Result
from sanic import Request
from co6co_sanic_ext .view_model import BaseView
from co6co_sanic_ext.api import add_routes
from sanic import Blueprint
from co6co_sanic_ext import sanics
from sanic import Sanic
import requests
from datetime import datetime
import random
import re
from typing import Dict, Optional, Union
from co6co_sanic_ext.utils.cors_utils import attach_cors


class BoxService:
    """相机会话管理类，用于与摄像头设备进行交互"""

    def __init__(self, ip_address: str, username: str, password: str,
                 device_ip: str = "192.168.0.50", device_port: int = 8080):
        """
        初始化相机会话

        Args:
            ip_address: 摄像头IP地址
            username: 登录用户名
            password: 登录密码
            device_ip: 设备内部IP地址（默认为192.168.0.50）
            device_port: 设备端口（默认为8080）
        """
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.device_ip = device_ip
        self.device_port = device_port
        self.session_id = None
        self.session = requests.Session()  # 持久会话管理
        self.agent = "mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/137.0.0.0 safari/537.36"
        # 配置默认请求头 - 修正User-Agent与实际请求一致
        self._default_headers = {
            "Content-Type": "text/plain",
            "Accept": "text/plain, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"http://{self.ip_address}/",  # 修正Referer与实际请求一致
            "Accept-Language": "zh-CN",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": self.agent,
            "Connection": "Keep-Alive",
            "Cache-Control": "no-cache",
            "Cookie": f"username={self.username}; password={self.password}"
        }

    def get_session_id(self) -> Optional[str]:
        """
        获取会话ID（登录认证）
        header 不是标准的,只能通过socket 读取
        HTTP/1.0 200 OK
        Date: Fri, 27 Jun 2025 01:46:28 GMT
        Server: HqBoa/0.94.13
        Connection: close
        [APP] enter common thread entry...
        Set-Cookie:sessionId=NWLRBBMQBH; path=/;
        Content-Type: text/plain;charset=utf-8

        Returns:
            成功返回sessionId，失败返回None
        """
        url = f"http://{self.ip_address}/cgi-bin/cgi_server.out"
        headers = self._get_login_headers()
        request_body = self._build_login_request()

        try:
            response = self.session.post(url, headers=headers, data=request_body)
            self._handle_response_status(response)

            print(response.text, response.headers, )

            if 'Set-Cookie' in response.headers:
                cookie_header = response.headers['Set-Cookie']
                self.session_id = self._extract_session_id(cookie_header)
                return self.session_id
            else:
                print("错误：响应中未包含Set-Cookie头，无法获取sessionId")
        except requests.RequestException as e:
            print(f"网络请求异常: {str(e)}")
        except Exception as e:
            print(f"处理响应异常: {str(e)}")

        return None

    def get_record_info(self, channel_id: int = 0,
                        start_time: Optional[str] = None,
                        record_sec: int = 3600) -> Optional[Dict]:
        """
        获取指定通道的录像信息

        Args:
            channel_id: 通道ID（默认0）
            start_time: 开始时间（格式YYYY-MM-DD HH:MM:SS，默认当前时间）
            record_sec: 录像时长（秒，默认3600）

        Returns:
            包含录像信息的字典，失败返回None
        """
        # if not self.session_id:
        #    print("错误：未获取到sessionId，请先调用get_session_id()")
        #    self.session_id = ""
        #    # return None

        url = f"http://{self.ip_address}/cgi-bin/cgi_server.out"
        headers = self._get_record_headers()
        request_body = self._build_record_request(channel_id, start_time, record_sec)

        try:
            response = self.session.post(url, headers=headers, data=request_body)
            self._handle_response_status(response)
            return self._parse_record_response(response.text), None
        except requests.RequestException as e:
            print(f"网络请求异常: {str(e)}")
            return None, str(e)
        except Exception as e:
            print(f"解析响应异常: {str(e)}")
            return None, str(e)

    def get_record_info2(self, channel_id: int = 0,
                         start_time: Optional[str] = None,
                         record_sec: int = 3600) -> Optional[Dict]:
        """
        获取指定通道的录像信息

        Args:
            channel_id: 通道ID（默认0）
            start_time: 开始时间（格式YYYY-MM-DD 00:00:00，默认当前时间）
            record_sec: 录像时长（秒，默认3600）

        Returns:
            包含录像信息的字典，失败返回None
        """
        # if not self.session_id:
        #    print("错误：未获取到sessionId，请先调用get_session_id()")
        #    self.session_id = ""
        #    # return None

        url = f"http://{self.ip_address}/cgi-bin/cgi_server.out"
        headers = self._get_record_headers()
        request_body = self._build_record_request(channel_id, start_time, record_sec, "get_record_info2")

        try:
            response = self.session.post(url, headers=headers, data=request_body)
            self._handle_response_status(response)
            return self._parse_record_response(response.text), None
        except requests.RequestException as e:
            print(f"网络请求异常: {str(e)}")
            return None, str(e)
        except Exception as e:
            print(f"解析响应异常: {str(e)}")
            return None, str(e)

    def _get_login_headers(self) -> Dict:
        """生成登录请求头"""
        return {
            **self._default_headers,
            "Host": self.ip_address,
            "Origin": f"http://{self.ip_address}",
            "Referer": f"http://{self.ip_address}/"  # 修正Referer与实际请求一致
        }

    def _get_record_headers(self) -> Dict:
        """生成获取录像信息的请求头"""
        return {
            **self._default_headers,
            "Host": self.ip_address,
            "Origin": f"http://{self.ip_address}",
            "Referer": f"http://{self.ip_address}/Video_Download.htm",
            "Cookie": f"username={self.username}; password={self.password}; sessionId={self.session_id}"
        }

    def _build_login_request(self) -> str:
        """构建登录请求体 - 修正IP地址和User-Agent"""
        content = [
            f"HL-CAM/1.0.0 command {self.device_ip}:{self.device_port}-udp",
            f"request-tag:{self._generate_random_tag()}",
            "content-length:0",
            "command:login",
            f"user-name:{self.username}",
            f"user-password:{self.password}",
            f"user-agent:{self.agent}",
            f"data-time:{self._get_current_datetime()}",
            "content-type:ini-fields",
            "\r\n"
        ]
        return "\r\n".join(content)

    def _build_record_request(self, channel_id: int,
                              start_time: Optional[str],
                              record_sec: int, comm="get_record_info") -> str:
        """
        构建获取录像信息的请求体 - 修正IP地址和User-Agent

        Args:
            channel_id: 通道ID
            start_time: 开始时间
            record_sec: 录像时长（秒）
        """
        if not start_time:
            # 默认为当前时间，与实际请求格式一致
            start_time = self._get_current_datetime()

        content = [
            f'HL-CAM/1.0.0 get-args {self.device_ip}:{self.device_port}-udp',
            f'request-tag:{self._generate_random_tag()}',
            'content-length:63',
            'user-agent:mozilla/5.0 (windows nt 10.0; wow64; trident/7.0; .net4.0c; .net4.0e; .net clr 2.0.50727; .net clr 3.0.30729; .net clr 3.5.30729; tablet pc 2.0; rv:11.0) like gecko',
            f'user-name:{self.username}',
            f'user-password:{self.password}',
            f'command:{comm}',
            f'user-auth:{self.username}',
            f'data-time:{self._get_current_datetime()}',
            'content-type:ini-fields',
            '\r\n',
            f'channel_id:{channel_id}',
            f'start_time:{start_time}',
            f'record_sec:{record_sec}'
        ]
        return "\r\n".join(content)

    def _parse_record_response(self, response_text: str) -> Dict:
        """
        解析录像信息响应 - 增强解析逻辑

        Args:
            response_text: 原始响应文本

        Returns:
            解析后的字典数据
        """
        response_data = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # 跳过协议标识行
        if lines and lines[0].startswith('HL-CAM/1.0.0 response'):
            lines = lines[1:]

        in_header = True
        for line in lines:
            if in_header:
                # 解析响应头字段
                if line.startswith('status-code:'):
                    response_data['status_code'] = line.split(':', 1)[1].strip()
                elif line.startswith('date-time:'):
                    response_data['date_time'] = line.split(':', 1)[1].strip()
                elif line.startswith('content-type:'):
                    response_data['content_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('response-tag:'):
                    response_data['response_tag'] = line.split(':', 1)[1].strip()
                    in_header = False  # 响应头解析完毕
            else:
                # 解析响应体键值对 - 支持更灵活的解析
                if ':' in line:
                    key, value = line.split(':', 1)
                    response_data[key.strip()] = value.strip()
                elif '=' in line:
                    key, value = line.split('=', 1)
                    response_data[key.strip()] = value.strip()

        return response_data

    def _generate_random_tag(self) -> str:
        """生成6位随机请求标签 - 与实际请求格式一致"""
        return str(random.randint(100000, 129999))

    def _get_current_datetime(self) -> str:
        """获取当前时间字符串（格式：YYYY-MM-DD HH:MM:SS）"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _extract_session_id(self, cookie_header: str) -> Optional[str]:
        """
        从Set-Cookie头中提取sessionId

        Args:
            cookie_header: Set-Cookie字段值

        Returns:
            提取的sessionId，未找到返回None
        """
        match = re.search(r'sessionId=([^;]+)', cookie_header)
        return match.group(1) if match else None

    def _handle_response_status(self, response: requests.Response) -> None:
        """
        处理响应状态码

        Args:
            response: requests响应对象

        Raises:
            AssertionError: 当状态码非200时抛出
        """
        if response.status_code != 200:
            raise AssertionError(f"请求失败，状态码: {response.status_code}，响应内容: {response.text[:100]}")


_dev_api = Blueprint("Box_API")


class recordParm:
    ip: str
    user_name: str
    passwd: str
    channel_id: int
    start_time: str
    record_sec: int


class BoxView(BaseView):

    routePath = "/getRecordInfo"

    async def post(self, request: Request):
        """
        param:
        {
            ip:str,
            user_name:str,
            passwd:str,
            channel_id:int,  # 通道号:[0|1|2|3]
            start_time:str,  # 查询时间 2025-06-27 01:00:00
            record_sec:int  # 查询 x 秒的
        }
        参数示例:
        {
            "ip": "192.168.1.100",
            "user_name": "admin",
            "passwd": "admin",
            "channel_id": 0,
            "start_time": "2025-06-27 23:00:00",
            "record_sec": 3600
        } 
        """
        param = recordParm()
        param.__dict__.update(request.json)
        svc = BoxService(param.ip, param.user_name, param.passwd)
        data, msg = svc.get_record_info(param.channel_id, param.start_time, param.record_sec)
        if msg:
            return self.response_json(Result.fail(data, msg))
        else:
            return self.response_json(Result.success(data))


class Box2View(BaseView):

    routePath = "/getRecordInfo2"

    async def post(self, request: Request):
        """
        param:
        {
            ip: str,
            user_name: str,
            passwd: str,
            channel_id: int,  # 通道号:[0|1|2|3]
            start_time: str,  # 查询时间  eg.2025-06-27 01:00:00
            record_sec: int  # 查询 x 秒的,eg. 86400
        }

        参数示例:
        {
            "ip": "192.168.1.100",
            "user_name": "admin",
            "passwd": "admin",
            "channel_id": 0,
            "start_time": "2025-06-27 00:00:00",
            "record_sec": 86400
        }
        """
        param = recordParm()
        param.__dict__.update(request.json)
        svc = BoxService(param.ip, param.user_name, param.passwd)
        data, msg = svc.get_record_info2(param.channel_id, param.start_time, param.record_sec)
        if msg:
            return self.response_json(Result.fail(data, msg))
        else:
            return self.response_json(Result.success(data))


add_routes(_dev_api, BoxView, Box2View)
box_api = Blueprint.group(_dev_api, url_prefix="/box")
api = Blueprint.group(box_api, url_prefix="/api", version=1)


def init(app: Sanic, _: dict):
    """
    初始化
    """
    # log.warn("APP:",   id(app))
    attach_cors(app)
    app.blueprint(api)


def main(config: dict):
    sanics.startApp(config, init)


# 使用示例
if __name__ == "__main__":
    config = sanics.getConfig(sanics.getConfigFilder(__file__))
    main(config)
    """
    # 创建相机会话实例
    camera = BoxService(
        ip_address="192.168.1.100",
        username="admin",
        password="admin"
    )

    # 获取sessionId
    # session_id = camera.get_session_id()
    # if not session_id:
    #    print("初始化失败，无法继续操作")
    #    exit(1)
    # print(f"成功获取sessionId: {session_id}")

    # 获取录像信息（指定开始时间为当前时间）
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_info = camera.get_record_info(
        channel_id=0,
        start_time=current_time,
        record_sec=3600
    )

    if record_info:
        print("录像信息解析结果:")
        for key, value in record_info.items():
            print(f"  {key}: {value}")
    else:
        print("获取录像信息失败")
    current_time = datetime.now().strftime("%Y-%m-%d 00:00:00")
    record_info = camera.get_record_info2(
        channel_id=0,
        start_time=current_time,
        record_sec=3600
    )
    if record_info:
        print("录像信息解析结果:")
        for key, value in record_info.items():
            print(f"  {key}: {value}")
    else:
        print("获取录像信息失败")
    """
