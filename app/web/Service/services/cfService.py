import requests


class CfService():
    """
    Cloudflare API 服务类

    需要提供以下参数：
    - API_TOKEN: Cloudflare API 令牌
    - ZONE_ID: Cloudflare 区域 ID


    """

    def __init__(self, API_TOKEN, ZONE_ID, timeout: int = 1500):
        self.headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        self.urlApi = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records"
        self.timeout = timeout
        pass

    def create_dns_record(self, type: str, name: str, content: str, ttl=1, proxied=False, comment: str = ""):
        data = {
            "type": type,
            "name": name,
            "content": content,
            "ttl": ttl,
            "proxied": proxied,
            "comment": comment
        }
        response = requests.post(self.urlApi, headers=self. headers, json=data, timeout=self.timeout)
        return response.json()

    def delete_dns_record(self, record_id):
        response = requests.delete(f"{self.urlApi}/{record_id}", headers=self.headers, timeout=self.timeout)
        return response.json()

    def update_dns_record(self, record_id: str, type: str, name: str, content: str, ttl=1, proxied=False, comment: str = ""):
        data = {
            "type": type,
            "name": name,
            "content": content,
            "ttl": ttl,
            "proxied": proxied,
            "comment": comment
        }
        print(f"{self.urlApi}/{record_id}")
        print(self.headers, data)
        response = requests.patch(f"{self.urlApi}/{record_id}", headers=self.headers, json=data, timeout=self.timeout)
        return response.json()

    def get_dns_record(self, record_id):
        response = requests.get(f"{self.urlApi}/{record_id}", headers=self.headers, timeout=self.timeout)
        return response.json()

    def list_dns_records(self):
        response = requests.get(self.urlApi, headers=self.headers, timeout=self.timeout)
        return response.json()
