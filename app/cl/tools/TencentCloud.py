# -*- coding: utf-8 -*-
# 腾讯云API签名v3实现示例
# 本代码基于腾讯云API签名v3文档实现: https://cloud.tencent.com/document/product/213/30654
# 请严格按照文档说明使用，不建议随意修改签名相关代码

import os
import hashlib
import hmac
import json
import sys
import time
from datetime import datetime
from co6co.utils import network

if sys.version_info[0] <= 2:
    from httplib import HTTPSConnection
else:
    from http.client import HTTPSConnection
    
# 密钥信息从环境变量读取，需要提前在环境变量中设置 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY
# 使用环境变量方式可以避免密钥硬编码在代码中，提高安全性
# 生产环境建议使用更安全的密钥管理方案，如密钥管理系统(KMS)、容器密钥注入等
# 请参见：https://cloud.tencent.com/document/product/1278/85305
# 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取

from co6co.enums import Base_Enum 
import urllib.request
import urllib.error

def get_public_ip() -> str | None:
    """
    使用标准库获取外网 IP，无需安装 requests
    """
    try:
        req = urllib.request.Request(
            "https://ipinfo.io/ip",
            headers={"User-Agent": "Mozilla/5.0"}  # 有些服务会拦没有 UA 的请求
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception as e:
        print(f"❌ 获取外网 IP 失败: {e}")
        return None

class TencentCloudAction(Base_Enum):
    ModifyRecord = "ModifyRecord", 0
    DescribeDomainList = "DescribeDomainList", 1
    DescribeRecordList = "DescribeRecordList", 2
    DescribeRecordLineList = "DescribeRecordLineList", 3


class TencentCloudDNS:
    def __init__(self, secret_id: str, secret_key: str):
        self.secret_id = secret_id
        self.secret_key = secret_key 
    def sign(self,key, msg):
         return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def get_domain_list(self):
        resp_data = self._request(TencentCloudAction.DescribeDomainList)
        domain_list = resp_data["Response"]["DomainList"]
        return domain_list
    
    def modify_record(self, domain: str,recordType:str,  record_id: int, value: str):
        payload = f'{{"Domain": "{domain}", "RecordType": "{recordType}",  "RecordId": {record_id}, "Value": "{value}","RecordLine": "默认"}}'
        resp_data = self._request(TencentCloudAction.ModifyRecord, payload)
        return resp_data 

    def get_record_line_list(self, domain: str,DomainGrade  = "0"):
        """
        获取域名的记录线列表
        :param domain: 域名
        :param DomainGrade: 域名等级，0为普通域名，DP为专业域名
        :return: 记录线列表
        """
        if DomainGrade not in ["0","DP"]:
            raise ValueError("DomainGrade must be 0 or DP")

        resp_data = self._request(TencentCloudAction.DescribeRecordLineList, f'{{"Domain": "{domain}", "DomainGrade": "{DomainGrade}"}}')
        
        return resp_data
    def get_record_list(self, domain: str):
        resp_data = self._request(TencentCloudAction.DescribeRecordList, f'{{"Domain": "{domain}"}}')
        record_list = resp_data["Response"]["RecordList"]
        return record_list 
    def find_record_id(self, domain: str, recordName: str):
        """
        查找域名的记录ID
        :param domain: 域名
        :param recordName: 记录名称
        :return: 记录ID
        """
        record_list = self.get_record_list(domain)
        print(record_list)
        for record in record_list:
            if record["Name"] == recordName:
                return record["RecordId"]
        return None

    def _request(self, action:TencentCloudAction, payload:str="{}", token: str = ""):
        service = "dnspod"
        host = "dnspod.tencentcloudapi.com"
        region = ""
        version = "2021-03-23"
        action = action.key 
        params = json.loads(payload)
        endpoint = "https://dnspod.tencentcloudapi.com"
        algorithm = "TC3-HMAC-SHA256"
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

        # ************* 步骤 1：拼接规范请求串 *************
        http_request_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (
            ct,
            host,
            action.lower(),
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = (
            http_request_method
            + "\n"
            + canonical_uri
            + "\n"
            + canonical_querystring
            + "\n"
            + canonical_headers
            + "\n"
            + signed_headers
            + "\n"
            + hashed_request_payload
        )

        # ************* 步骤 2：拼接待签名字符串 *************
        credential_scope = date + "/" + service + "/" + "tc3_request"
        hashed_canonical_request = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()
        string_to_sign = (
            algorithm
            + "\n"
            + str(timestamp)
            + "\n"
            + credential_scope
            + "\n"
            + hashed_canonical_request
        )

        # ************* 步骤 3：计算签名 *************
        secret_date =self. sign(("TC3" +self.secret_key).encode("utf-8"), date)
        secret_service =self. sign(secret_date, service)
        secret_signing =self. sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # ************* 步骤 4：拼接 Authorization *************
        authorization = (
            algorithm
            + " "
            + "Credential="
            + secret_id
            + "/"
            + credential_scope
            + ", "
            + "SignedHeaders="
            + signed_headers
            + ", "
            + "Signature="
            + signature
        )

        # ************* 步骤 5：构造并发起请求 *************
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": timestamp,
            "X-TC-Version": version,
        }
        if region:
            headers["X-TC-Region"] = region
        if token:
            headers["X-TC-Token"] = token

        try:
            req = HTTPSConnection(host)
            req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
            resp = req.getresponse()
            resp_data = resp.read().decode("utf-8")
            resp_data = json.loads(resp_data)
            return resp_data
        except Exception as err:
            print(err)
 



if __name__ == "__main__":
    
    secret_id =  ""  # os.getenv("TENCENTCLOUD_SECRET_ID") 
    secret_key = ""  # os.getenv("TENCENTCLOUD_SECRET_KEY")
    dns = TencentCloudDNS(secret_id, secret_key)
    print(get_public_ip())
    ip=get_public_ip()
    if ip: 
        domain='ynlanbo.com'
        recordName="asset"
        record_id= dns.find_record_id(domain, recordName)
        if record_id:
            print(record_id)
            print( dns.modify_record(domain, "A",record_id, ip))
        else:
            print("未找到记录ID")  
    #network.get_local_ip()
    #print(record_id)
    
    
     


    