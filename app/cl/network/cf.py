import requests

# 替换为你的 Cloudflare API 令牌
API_TOKEN = "your_api_token"
# 替换为你的 Cloudflare 区域 ID
ZONE_ID = "your_zone_id"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}


def create_dns_record(record_type, name, content, ttl=1, proxied=False):
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records"
    data = {
        "type": record_type,
        "name": name,
        "content": content,
        "ttl": ttl,
        "proxied": proxied
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def delete_dns_record(record_id):
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records/{record_id}"
    response = requests.delete(url, headers=headers)
    return response.json()


def update_dns_record(record_id, record_type, name, content, ttl=1, proxied=False):
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records/{record_id}"
    data = {
        "type": record_type,
        "name": name,
        "content": content,
        "ttl": ttl,
        "proxied": proxied
    }
    response = requests.patch(url, headers=headers, json=data)
    return response.json()


def get_dns_record(record_id):
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records/{record_id}"
    response = requests.get(url, headers=headers)
    return response.json()


def list_dns_records():
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records"
    response = requests.get(url, headers=headers)
    return response.json()


if __name__ == "__main__":
    # 创建 DNS 记录示例
    create_response = create_dns_record("A", "example.com", "1.2.3.4")
    print("创建 DNS 记录响应:", create_response)

    # 列出 DNS 记录示例
    list_response = list_dns_records()
    print("列出 DNS 记录响应:", list_response)

    # 假设获取到了记录 ID
    if list_response["success"] and list_response["result"]:
        record_id = list_response["result"][0]["id"]

        # 获取 DNS 记录详情示例
        get_response = get_dns_record(record_id)
        print("获取 DNS 记录详情响应:", get_response)

        # 更新 DNS 记录示例
        update_response = update_dns_record(record_id, "A", "example.com", "5.6.7.8")
        print("更新 DNS 记录响应:", update_response)

        # 删除 DNS 记录示例
        delete_response = delete_dns_record(record_id)
        print("删除 DNS 记录响应:", delete_response)
