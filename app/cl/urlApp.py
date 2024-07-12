import urllib.parse
from co6co.utils import log

base_url = "http://wx.co6co.top:444/v1/api/wx/wx98f9d28a69632548/oauth"
params = {
    "backurl": "http://wx.co6co.top:444/xd/home.html",
}

# 使用urllib.parse.urlencode来编码参数
query_string = urllib.parse.urlencode(params)
log.succ("queryStr:", query_string)

# 构造完整的redirect_uri
redirect_uri = f"{base_url}?{query_string}"
log.succ("qllURL:", redirect_uri)
# 对redirect_uri进行URL编码，因为它是URL的一部分
encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
log.succ("qllURL ENcode:", encoded_redirect_uri)
# 构造完整的授权请求URL
auth_url = f"https://open.weixin.qq.com/connect/oauth2/authorize?appid=APPID&redirect_uri={
    encoded_redirect_uri}&response_type=code&scope=snsapi_userinfo&state=STATE#wechat_redirect"
log.succ("authod ENcode:", auth_url)


url = "http://localhost:5173/xd/home.html#/userInfo?123456asdf=4&a=1"
encoded_redirect_uri = urllib.parse.quote(url, safe='')
log.succ("qllURL ENcode:", encoded_redirect_uri)
