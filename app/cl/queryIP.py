import requests
# pip install geoip2
'''
from geoip2.database import Reader

# 指定GeoLite2 City数据库的路径
database_path = '/path/to/GeoLite2-City.mmdb'

# 创建Reader对象
reader = Reader(database_path)

# 使用IP地址查询位置信息
response = reader.city('8.8.8.8')  # 这里使用Google的公共DNS服务器作为示例

# 输出查询结果
print(f'City: {response.city.name}')
print(f'Region: {response.subdivisions.most_specific.name}')
print(f'Country: {response.country.name}')

# 关闭数据库连接
reader.close()
'''


def get_location_by_ip(ip, lang="zh-CN"):
    response = requests.get(f'http://ip-api.com/json/{ip}?lang={lang}')
    data = response.json()
    return data['city'], data['regionName'], data['country'], data['lat'], data["lon"]


# 示例

city, region, country, lat, lon = get_location_by_ip('116.54.34.230')
print(f'City: {city}, Region: {region}, Country: {country},{lat},{lon}')
