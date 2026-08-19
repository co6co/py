import requests

url = "https://XXX/web/v1/portal/uploadfile"
headers = {
    "Authorization": "3529ac8c-ef30-4dd7-ae12-cc34cb0ec153",
    "referer": "https://XXX",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
}
files = {
    "file": (
        "txt.jpeg",  # 文件名
        open("10MB.data", "rb"),  # 文件对象
        "image/jpeg",  # MIME 类型
    )
}

response = requests.post(url, headers=headers, files=files, timeout=1000)
print(response.json())
