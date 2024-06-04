from bs4 import BeautifulSoup
from co6co.utils import http
import requests
import webbrowser
url = "https://www.youtube.com/playlist?list=PLpPexpOJZjK8PFjwhCpKpCE9vU2MULT7W"
proxy = {'http': 'socks5://127.0.0.1:9666', 'https': 'socks5://127.0.0.1:9666'}
response = http.get(url, proxy=proxy, verify=False)

# lxml|lxml-xml|xml|html5lib
soup = BeautifulSoup(response.text, 'html.parser')
# 格式化输出
print(soup.prettify())
data = soup.select('#contents')
print(data)
