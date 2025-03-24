import requests


class CfService():
    def __init__(self, API_TOKEN, ZONE_ID):
        self.headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        self.urlApi = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records"
        pass

    def create_dns_record(self, record_type, name, content, ttl=1, proxied=False):
        data = {
            "type": record_type,
            "name": name,
            "content": content,
            "ttl": ttl,
            "proxied": proxied
        }
        response = requests.post(self.urlApi, headers=self. headers, json=data)
        return response.json()

    def delete_dns_record(self, record_id):
        response = requests.delete(f"{self.urlApi}/{record_id}", headers=self.headers)
        return response.json()

    def update_dns_record(self, record_id, record_type, name, content, ttl=1, proxied=False):
        data = {
            "type": record_type,
            "name": name,
            "content": content,
            "ttl": ttl,
            "proxied": proxied
        }
        response = requests.patch(f"{self.urlApi}/{record_id}", headers=self.headers, json=data)
        return response.json()

    def get_dns_record(self, record_id):
        response = requests.get(f"{self.urlApi}/{record_id}", headers=self.headers)
        return response.json()

    def list_dns_records(self):
        response = requests.get(self.urlApi, headers=self.headers)
        return response.json()


if __name__ == "__main__":
    # 替换为你的 Cloudflare API 令牌
    API_TOKEN = "RHqNM_zZ1ORKXXXXXXXXXXXXXXXXXXXXXDMAfjG_"
    # 替换为你的 Cloudflare 区域 ID
    ZONE_ID = "fcc0ed5000000000000000000000fb6ba88"
    service = CfService(API_TOKEN, ZONE_ID)
    # 创建 DNS 记录示例
    create_response = service. create_dns_record("A", "example.com", "1.2.3.4")
    # print("创建 DNS 记录响应:", create_response)

    # 列出 DNS 记录示例
    list_response = service. list_dns_records()
    print("列出 DNS 记录响应:", list_response)

    # 假设获取到了记录 ID
    if list_response["success"] and list_response["result"]:
        record_id = list_response["result"][0]["id"]

        # 获取 DNS 记录详情示例
        get_response = service. get_dns_record(record_id)
        print(f"获取 DNS 记录<{record_id}>详情响应:", get_response)

        # 更新 DNS 记录示例
        update_response = service.update_dns_record(record_id, "A", "example.com", "5.6.7.8")
        print("更新 DNS 记录响应:", update_response)

        # 删除 DNS 记录示例
        delete_response = service. delete_dns_record(record_id)
        # print("删除 DNS 记录响应:", delete_response)
